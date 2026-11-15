"""What does the leaderboard's ordering of our own submissions say about the TEST set?

Every submission since sub033 changes the kept set by a handful of masks, and the board orders
them by true score. Locally every candidate carries a label, so we know exactly which kinds of
candidate pay; on test we know the same candidates' properties but not their labels. Putting the
two together turns the ordering into evidence about the test distribution:

  1. break-even by property bin (train).  The metric pays v = copy-weighted IoU sum for a kept
     candidate and charges c/2 for it, so a candidate is worth keeping iff v > c/2 * PQ. Binned by
     contrast, size, crowding and radius, this says which filament kinds a rule may safely admit.
  2. what each arm actually admitted (train vs test).  The same rule is applied to both pools; the
     property profile of the admitted set says whether the arm met the same population on test as
     on train. A rule that admits crowded faint candidates on train and bright isolated ones on
     test is not the same experiment.
  3. the arms' board direction.  Combined with (1) and (2): an arm whose admitted set is pure on
     train and sank on the board admitted a population that behaves differently on test.
  4. implied test precision.  The denominator identity D = 1/2 * sum_kept c + 1/2 * G makes the
     kept set's match rate recoverable from the displayed score under stated assumptions about the
     test set's annotator count and ground-truth density, and it can be compared with the local
     match rate at the same gate.

Assumption stated once: the test set's copies-per-image and GT rows per image are taken from the
training set (1.635 and 11.66); both are reported with a sensitivity range, because neither is
observable. Everything else is measured.

Usage: python src/eda/test_set_inference.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEL = ROOT / "outputs" / "selector"
GATE = (0.40, 0.60)
PROPS = ["img_contrast", "log10_area", "n_within_200px", "r_norm", "width_proxy", "skel_len",
         "sem_prob_mean", "sem_contrast", "conf", "eccentricity", "nn_dist", "coh_in"]


def log(*a):
    print(*a, flush=True)


def bundle_scores(bundle, X):
    if bundle.get("bagged"):
        raw = np.mean([c.predict_proba(X)[:, 1] * np.clip(r.predict(X), 0.0, None)
                       for c, r in bundle["pairs"]], axis=0)
    else:
        raw = bundle["model"].predict_proba(X)[:, 1]
        if bundle.get("iou_model") is not None:
            raw = raw * np.clip(bundle["iou_model"].predict(X), 0.0, None)
    ref = bundle["two_stage_cdf"]
    return np.searchsorted(ref, raw, side="right") / len(ref)


def cascade_mask(pool, p, feats, train, conf=GATE[0], pmin=GATE[1], q=0.85):
    """E138 band cascade admissions, fitted on the training pool's band (same code path as the
    submissions: a band-only classifier, top (1-q) of the test band admitted)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    band_tr = train[(train["conf"] >= conf) & (train["p"] >= 0.40) & (train["p"] < pmin)]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=25, l2_regularization=1.0, random_state=42)
    m.fit(band_tr[feats].to_numpy(np.float64), band_tr["y"].to_numpy())
    band = (pool["conf"] >= conf).to_numpy() & (p >= 0.40) & (p < pmin)
    if not band.any():
        return band
    s = np.full(len(pool), -1.0)
    s[band] = m.predict_proba(pool.loc[band, feats].to_numpy(np.float64))[:, 1]
    return band & (s >= np.quantile(s[band], q))


def profile(df, cols=PROPS):
    return {c: float(df[c].median()) for c in cols} | {"n": int(len(df))}


def main():
    ref = joblib.load(SEL / "selector_v5_shrunk_t020pox.joblib")
    feats = list(ref["features"])

    test = pd.read_parquet(OUT / "test_pool_t020pox.parquet")
    train = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    train = train.merge(pd.read_parquet(OUT / "extra_features_t020.parquet")[["cand_id"] + EXTRA],
                        on="cand_id", how="left")
    train = train.merge(pd.read_parquet(OUT / "t020pox_oof_scores.parquet"), on="cand_id", how="left")
    train = train.merge(pd.read_parquet(OUT / "bag5_oof_scores.parquet").rename(columns={"p": "p_bag"}),
                        on="cand_id", how="left")
    Xtr = train[feats].to_numpy(np.float64)
    Xte = test[feats].to_numpy(np.float64)
    p_te = {"single": test["p_bundle"].to_numpy(),
            "bag": bundle_scores(joblib.load(SEL / "selector_t020pox_bag5.joblib"), Xte)}
    p_tr = {"single": train["p"].to_numpy(), "bag": train["p_bag"].to_numpy()}
    res = {}

    # ---- 1. the metric's break-even, by property bin, on the training pool -------------------
    pq0 = 0.4421
    train["v"] = train["iou_sum"]
    train["breakeven"] = 0.5 * train["n_copies"] * pq0
    kept_tr = (train["conf"] >= GATE[0]) & (train["p"] >= GATE[1])
    band_tr = (train["conf"] >= GATE[0]) & (train["p"] >= 0.40) & (train["p"] < GATE[1])
    log("\n=== 1. What a candidate is worth, by property bin (training pool, labelled) ===")
    log("    v = copy-weighted IoU sum the metric pays; break-even = 0.5*c*PQ (mean 0.362).")
    log("    'band' = the marginal population every admitting rule draws from (p 0.40-0.60).\n")
    bins = {
        "contrast (img_contrast)": ("img_contrast", [-np.inf, 0.05, 0.10, 0.16, np.inf]),
        "size (log10 area)": ("log10_area", [-np.inf, 2.7, 3.0, 3.4, np.inf]),
        "crowding (n within 200px)": ("n_within_200px", [-0.5, 0.5, 1.5, 3.5, np.inf]),
        "radius (r_norm)": ("r_norm", [-np.inf, 0.3, 0.6, 0.8, np.inf]),
    }
    table1 = {}
    for name, (col, edges) in bins.items():
        lab = pd.cut(train[col], edges)
        rows = []
        for key, idx in train.groupby(lab, observed=True).groups.items():
            sub = train.loc[idx]
            sb = sub[band_tr.loc[idx]]
            rows.append(dict(bin=str(key), n=len(sub), n_band=len(sb),
                             kept_share=float(kept_tr.loc[idx].mean()),
                             band_truth=float(sb["y"].mean()) if len(sb) else np.nan,
                             band_v=float(sb["v"].mean()) if len(sb) else np.nan,
                             band_breakeven=float(sb["breakeven"].mean()) if len(sb) else np.nan))
        t = pd.DataFrame(rows)
        t["band_pays"] = t["band_v"] - t["band_breakeven"]
        table1[name] = t.to_dict("records")
        log(f"  {name}")
        log("   " + t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res["breakeven_by_bin"] = table1

    # ---- 2/3. what each arm admitted, train vs test, against its board direction --------------
    log("\n=== 2. What each arm admitted or dropped, train vs test ===")
    board = {r["file"].split("_")[0]: i + 1
             for i, r in enumerate(json.load(open(OUT / "board_ordering.json"))["order"])}
    base_te = (test["conf"] >= GATE[0]).to_numpy() & (p_te["single"] >= GATE[1])
    base_tr = kept_tr.to_numpy()
    arms = {
        "sub037 cascade q0.85": (
            base_te | cascade_mask(test, p_te["single"], feats, train),
            base_tr | cascade_mask(train, p_tr["single"], feats, train)),
        "sub039 p 0.55": ((test["conf"] >= 0.40).to_numpy() & (p_te["single"] >= 0.55),
                          (train["conf"] >= 0.40).to_numpy() & (p_tr["single"] >= 0.55)),
        "sub040 conf 0.45": ((test["conf"] >= 0.45).to_numpy() & (p_te["single"] >= 0.60),
                             (train["conf"] >= 0.45).to_numpy() & (p_tr["single"] >= 0.60)),
        "sub036 bagging x5": ((test["conf"] >= 0.40).to_numpy() & (p_te["bag"] >= 0.60),
                              (train["conf"] >= 0.40).to_numpy() & (p_tr["bag"] >= 0.60)),
    }
    out2 = {}
    for name, (keep_te, keep_tr) in arms.items():
        sub = name.split()[0]
        add_te, drop_te = test[keep_te & ~base_te], test[~keep_te & base_te]
        add_tr, drop_tr = train[keep_tr & ~base_tr], train[~keep_tr & base_tr]
        d = dict(board_rank=board.get(sub), n_add_test=len(add_te), n_drop_test=len(drop_te),
                 n_add_train=len(add_tr), n_drop_train=len(drop_tr),
                 add_train_truth=float(add_tr["y"].mean()) if len(add_tr) else np.nan,
                 add_train_v=float(add_tr["v"].mean()) if len(add_tr) else np.nan,
                 add_train_breakeven=float(add_tr["breakeven"].mean()) if len(add_tr) else np.nan,
                 drop_train_truth=float(drop_tr["y"].mean()) if len(drop_tr) else np.nan,
                 drop_train_v=float(drop_tr["v"].mean()) if len(drop_tr) else np.nan)
        d["add_profile_train"] = profile(add_tr) if len(add_tr) else {}
        d["add_profile_test"] = profile(add_te) if len(add_te) else {}
        out2[name] = d
        log(f"\n  {name}  (board rank {d['board_rank']})")
        log(f"    admitted: train {len(add_tr)}, test {len(add_te)} | dropped: train {len(drop_tr)}, test {len(drop_te)}")
        if len(add_tr):
            log(f"    admitted on train: truth {d['add_train_truth']:.3f}, v {d['add_train_v']:.3f} "
                f"vs break-even {d['add_train_breakeven']:.3f} -> "
                f"{'pays' if d['add_train_v'] > d['add_train_breakeven'] else 'costs'}")
        if len(drop_tr):
            log(f"    dropped  on train: truth {d['drop_train_truth']:.3f}, v {d['drop_train_v']:.3f}")
        if len(add_tr) and len(add_te):
            comp = pd.DataFrame({"train": pd.Series(d["add_profile_train"]),
                                 "test": pd.Series(d["add_profile_test"])}).drop("n")
            comp["test/train"] = comp["test"] / comp["train"].replace(0, np.nan)
            log("    admitted population, median properties:")
            log("      " + comp.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n      "))
    res["arms"] = out2

    # ---- 4. the pools themselves, and the implied test precision -----------------------------
    log("\n=== 3. Candidate pools: train vs test ===")
    comp = pd.DataFrame({"train_all": train[PROPS].median(), "test_all": test[PROPS].median(),
                         "train_kept": train.loc[base_tr, PROPS].median(),
                         "test_kept": test.loc[base_te, PROPS].median()})
    comp["kept_ratio"] = comp["test_kept"] / comp["train_kept"].replace(0, np.nan)
    log(comp.to_string(float_format=lambda x: f"{x:.3f}"))
    res["pool_medians"] = comp.to_dict()
    per_img = dict(train_cands=len(train) / train["stem"].nunique(),
                   test_cands=len(test) / test["stem"].nunique(),
                   train_kept=int(base_tr.sum()) / train["stem"].nunique(),
                   test_kept=int(base_te.sum()) / test["stem"].nunique())
    log(f"\n  candidates per image: train {per_img['train_cands']:.2f}, test {per_img['test_cands']:.2f}"
        f" | kept per image: train {per_img['train_kept']:.2f}, test {per_img['test_kept']:.2f}")
    res["per_image"] = per_img

    log("\n=== 4. Implied test precision from the displayed score ===")
    log("    D = 0.5*sum_kept c + 0.5*G  (exact, ledger 34); U = SQ * TP; PQ = U/D.")
    tp_tr = 4802; fp_tr = 1733; fn_tr = 3393
    sq_tr = 0.4422 * (tp_tr + 0.5 * fp_tr + 0.5 * fn_tr) / tp_tr
    log(f"    local at the same gate: kept {int(base_tr.sum())}, TP {tp_tr}, FP {fp_tr}, "
        f"precision {tp_tr/(tp_tr+fp_tr):.3f}, SQ {sq_tr:.3f}")
    rows = []
    for cbar in (1.4, 1.635, 1.9):
        for gdens in (10.0, 11.66, 13.0):
            for pq_true in (0.395, 0.400, 0.405):
                kept = int(base_te.sum()); n_img = test["stem"].nunique()
                D = 0.5 * kept * cbar + 0.5 * gdens * n_img
                U = pq_true * D
                for sq in (sq_tr,):
                    tp = U / sq
                    prec = tp / (kept * cbar)
                    rows.append(dict(c=cbar, gt_per_img=gdens, pq=pq_true, TP=tp,
                                     precision=prec, ratio_vs_local=prec / (tp_tr / (tp_tr + fp_tr))))
    t4 = pd.DataFrame(rows)
    log(t4.groupby(["c", "gt_per_img"])["precision"].agg(["min", "max"]).to_string(
        float_format=lambda x: f"{x:.3f}"))
    log(f"    (local precision at this gate: {tp_tr/(tp_tr+fp_tr):.3f}; SQ assumed equal to local)")
    res["implied_precision"] = t4.to_dict("records")

    json.dump(res, open(OUT / "test_set_inference.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'test_set_inference.json'}")


if __name__ == "__main__":
    main()
