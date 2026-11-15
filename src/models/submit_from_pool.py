"""Write a submission from a dumped test candidate pool, so selector-side variants cost seconds.

submit_v2.py --dump-pool writes every fused test candidate with the features the selector reads
and its shrunk mask. This applies a selector bundle (the plain two-stage bundle, or a bagged one
from fit_selector_bagged.py) and a keep gate to that pool, and writes the CSV the same way
submit_v2 does: masks in sorted image order, masks under 64 px dropped. Rebuilding sub033 from
the pool must reproduce its CSV byte for byte; `--check` compares against a reference file.

Usage: python src/models/submit_from_pool.py --pool outputs/eda/test_pool_t020pox.parquet \
         --selector outputs/selector/selector_v5_shrunk_t020pox.joblib --conf 0.40 --p-min 0.60 \
         --out outputs/submissions/subXXX.csv [--check outputs/submissions/sub033_...csv]
"""
import argparse
import filecmp
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle, write_submission  # noqa: E402


def scores(bundle, X):
    if bundle.get("bagged"):
        raw = np.mean([c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0.0, None)
                       for c, r in bundle["pairs"]], axis=0)
    else:
        raw = bundle["model"].predict_proba(X)[:, 1]
        if bundle.get("iou_model") is not None:
            raw = raw * np.clip(bundle["iou_model"].predict(X), 0.0, None)
    ref = bundle["two_stage_cdf"]
    return np.searchsorted(ref, raw, side="right") / len(ref)


# The detector lane is a DIFFERENT SIZE in fitting and in deployment, and that is structural,
# not a modelling slip: for a training image in fold F only model F is honest, so 74% of local
# candidates carry exactly one detector vote (det_conf_std is literally 0.000 for them), while
# deployment votes all five and 49% of test candidates carry five. Measured shift (E201,
# ledger 41): n_det_models median 1 -> 4, det_conf 0.290 -> 0.371, det_conf_std 0.000 -> 0.047.
# A gradient-boosted tree puts its splits where the FITTING distribution varies; under that shift
# the test candidates pile up on one side of those splits and the feature stops discriminating
# WITHIN the test set, which is the only thing PQ depends on. Rank-based transport onto the local
# marginal restores it. It is monotone, so it never reorders candidates by the feature itself.
TRANSPORT = ["det_conf", "det_conf_max", "det_conf_mean", "det_conf_std", "n_det_models"]


def transport(pool, feats, ref_pool):
    """Map each listed feature's test values onto the local marginal by rank."""
    out, moved = pool.copy(), []
    for f in TRANSPORT:
        if f not in feats or f not in ref_pool.columns:
            continue
        src = pool[f].to_numpy(float)
        ref = np.sort(ref_pool[f].to_numpy(float))
        q = src.argsort().argsort() / max(len(src) - 1, 1)          # empirical rank in [0,1]
        out[f] = np.quantile(ref, q)
        moved.append(f"{f}: {np.median(src):.3f} -> {np.median(out[f]):.3f}")
    return out, moved


def cascade_admit(pool, p, feats, a):
    """E138 marginal-band cascade (src/eda/band_cascade.py, +0.0003 TIED over five folds).

    A second classifier, fitted only on the training pool's band candidates (conf >= gate,
    out-of-fold p in [0.40, gate p)), re-ranks the test band, and the band candidates at or above
    the `cascade_q` quantile of its score within the test band are added to the keep set. The five
    folds of E138 chose quantiles 0.80/0.95/0.85/0.90/0.85; their median, 0.85, is the default.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    sys.path.insert(0, str(ROOT / "src" / "models"))
    from extra_features import EXTRA
    tr = pd.read_parquet(ROOT / "outputs/eda/fused_pool_all_blob5_t020_labelled.parquet")
    tr = tr.merge(pd.read_parquet(ROOT / "outputs/eda/extra_features_t020.parquet")[["cand_id"] + EXTRA],
                  on="cand_id", how="left")
    oof = pd.read_parquet(ROOT / a.cascade_scores)
    tr = tr.merge(oof, on="cand_id", how="left", validate="one_to_one")
    if tr["p"].isna().any():
        raise ValueError("Cascade OOF scores do not cover the training pool")
    band_tr = tr[(tr["conf"] >= a.conf) & (tr["p"] >= 0.40) & (tr["p"] < a.p_min)]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=25, l2_regularization=1.0, random_state=42)
    m.fit(band_tr[feats].to_numpy(np.float64), band_tr["y"].to_numpy())
    band = (pool["conf"] >= a.conf).to_numpy() & (p >= 0.40) & (p < a.p_min)
    if not band.any():
        return band
    s = np.full(len(pool), -1.0)
    s[band] = m.predict_proba(pool.loc[band, feats].to_numpy(np.float64))[:, 1]
    cut = np.quantile(s[band], a.cascade_q)
    admit = band & (s >= cut)
    print(f"cascade: train band {len(band_tr)}, test band {int(band.sum())}, admitted {int(admit.sum())}")
    return admit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--selector", required=True)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--p-min", type=float, default=0.60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="")
    ap.add_argument("--cascade-scores", default="outputs/eda/t020pox_oof_scores.parquet",
                    help="OOF scores from the first-stage selector used to define the training band")
    ap.add_argument("--transport", default="",
                    help="labelled training pool to transport the detector-lane features onto "
                         "(ledger 41). Empty = deployed behaviour.")
    ap.add_argument("--cascade-q", type=float, default=1.0,
                    help="admit the E138 band cascade's top (1 - q) of the test band; 1.0 = off")
    ap.add_argument("--p-min-busy", type=float, default=None,
                    help="regime gate (E212, ledger 52): this p threshold in frames whose hf exceeds "
                         "--hf-split, --p-min elsewhere. Unset = deployed behaviour.")
    ap.add_argument("--hf-split", type=float, default=587.0,
                    help="training median of the disc hf (outputs/eda/image_quality.parquet)")
    a = ap.parse_args()
    pool = pd.read_parquet(ROOT / a.pool)
    bundle = joblib.load(ROOT / a.selector)
    if a.transport:
        pool, moved = transport(pool, bundle["features"], pd.read_parquet(ROOT / a.transport))
        print("transported: " + "; ".join(moved))
    p = scores(bundle, pool[bundle["features"]].to_numpy(np.float64))
    p_min = np.full(len(pool), a.p_min)
    if a.p_min_busy is not None:
        # hf is read from the test image itself (outputs/eda/_image_quality.py), never from labels
        q = pd.read_parquet(ROOT / "outputs/eda/image_quality.parquet")
        hf = pool["stem"].map(q[q["split"] == "test"].set_index("stem")["hf"])
        assert hf.notna().all(), "hf missing for some test frames"
        busy = (hf > a.hf_split).to_numpy()
        p_min[busy] = a.p_min_busy
        print(f"regime gate: p >= {a.p_min_busy} in {pool.loc[busy, 'stem'].nunique()} busy frames "
              f"(hf > {a.hf_split:.0f}), p >= {a.p_min} elsewhere")
    keep = (pool["conf"] >= a.conf).to_numpy() & (p >= p_min)
    if a.cascade_q < 1.0:
        keep |= cascade_admit(pool, p, bundle["features"], a)
    stems = sorted(pool["stem"].unique())
    preds = {}
    for stem in stems:
        idx = np.flatnonzero(keep & (pool["stem"] == stem).to_numpy())
        masks = [m.astype(np.uint8) for m in (decode_rle(pool["rle_shrunk"].iloc[i]).astype(bool) for i in idx)
                 if m.sum() >= 64]
        if masks:
            preds[stem] = masks
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = write_submission(preds, out)
    print(f"kept {int(keep.sum())} candidates -> {len(df)} rows over {len(preds)} images -> {a.out}")
    if a.check:
        same = filecmp.cmp(out, ROOT / a.check, shallow=False)
        print(f"byte-identical to {a.check}: {same}")
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
