"""E202 -- what does the 1-detector / 5-detector mismatch actually COST?

Ledger 41: the local pool is built out of fold, so 74% of its candidates carry exactly one
detector vote (det_conf_std is 0.000 for them) while deployment votes five and 49% of test
candidates carry five. Transporting the test features onto the local marginal is one possible
correction, but it cannot be validated locally -- on the local pool the transport is the identity
by construction.

The mismatch can be validated, by running it BACKWARDS. Fit the selector out of fold on the
untransported local pool exactly as deployed, then score the held-out fold twice:

  as fitted      the candidate's own features            -> the number we always quote
  deployment-like the same candidates with the detector features transported onto the TEST
                 marginal, i.e. what the fitted model sees when it is shipped

The gap between them is the price the mismatch charges at deployment, measured with the exact
scorer on data where we have ground truth. If it is ~0 the transport arm is not worth a slot; if
it is large the arm has a measured mechanism behind it, which is more than any recent arm had.

Usage: python src/eda/mismatch_cost.py [--bag 1]
"""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402
from fit_selector_bagged import fit_bag, bag_score  # noqa: E402
from submit_from_pool import TRANSPORT  # noqa: E402

OUT = ROOT / "outputs" / "eda"
GATE = (0.40, 0.60)


def to_marginal(df, feats, ref, cols):
    out = df.copy()
    for f in cols:
        if f not in feats or f not in ref.columns:
            continue
        src = df[f].to_numpy(float)
        q = src.argsort().argsort() / max(len(src) - 1, 1)
        out[f] = np.quantile(np.sort(ref[f].to_numpy(float)), q)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", type=int, default=1); a = ap.parse_args()
    feats = list(joblib.load(ROOT / "outputs/selector/selector_v5_shrunk_t020pox.joblib")["features"])
    pool = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    pool = pool.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA], on="cand_id")
    test = pd.read_parquet(OUT / "test_pool_t020pox.parquet")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    pool["fold"] = pool["stem"].map(fold_of).astype(int)
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(pool["stem"]))]
    stems = np.array(sorted(set(pool["stem"])))
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()

    shifted = to_marginal(pool, feats, test, TRANSPORT)     # local made to LOOK like deployment
    print("yerel -> test marjinaline tasindi:")
    for f in TRANSPORT:
        if f in feats:
            print(f"  {f:15s} {pool[f].median():.3f} -> {shifted[f].median():.3f} "
                  f"(test {test[f].median():.3f})")
    rng = np.random.default_rng(0)
    Xa, Xb = pool[feats].to_numpy(np.float64), shifted[feats].to_numpy(np.float64)
    raw = {"uydurulan gibi": np.zeros(len(pool)), "dagitim gibi": np.zeros(len(pool))}
    for F in range(5):
        tr = (pool["fold"] != F).to_numpy(); te = ~tr
        pairs = fit_bag(pool[tr], feats, a.bag, rng if a.bag > 1 else None)
        inner = bag_score(pairs, Xa[tr])                 # CDF from the fitting distribution
        ref = np.sort(inner)
        for k, X in (("uydurulan gibi", Xa), ("dagitim gibi", Xb)):
            raw[k][te] = np.searchsorted(ref, bag_score(pairs, X[te]), side="right") / len(ref)
        print(f"  fold {F} bitti", flush=True)
    res = {}
    for k, p in raw.items():
        keep = (pool["conf"] >= GATE[0]).to_numpy() & (p >= GATE[1])
        st = per_image(pool[keep][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
        val = float(pq(st["tp"].sum(), st["fp"].sum(), st["fn"].sum(), st["iou"].sum()))
        res[k] = dict(pq=val, kept=int(keep.sum()))
        print(f"{k:16s} tutulan {int(keep.sum()):5d}  PQ {val:.4f}")
    d = res["dagitim gibi"]["pq"] - res["uydurulan gibi"]["pq"]
    print(f"\nUYUMSUZLUGUN BEDELI: {d:+.4f} PQ, tutulan {res['dagitim gibi']['kept']-res['uydurulan gibi']['kept']:+d}")
    res["cost"] = d
    json.dump(res, open(OUT / "mismatch_cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
