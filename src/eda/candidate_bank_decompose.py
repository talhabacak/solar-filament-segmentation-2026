"""E096 §10e -- decompose the legal-oracle ceiling into its four ingredients,
each measured by realised PQ after the legal (non-overlap) pass.

The legal oracle (candidate_bank_oracle.py) differs from the production path
in FOUR ways, and the earlier capture arms only ever varied one of them:
  choice    which variant per matched cluster (oracle-best vs vote50)
  rescue    clusters whose vote50 misses IoU 0.5 but some variant hits
  order     the de-overlap order: production confidence vs GT IoU
  post      a GT check AFTER carving: a mask that falls to IoU <= 0.5 once
            its overlap with earlier masks is removed is dropped (the oracle
            script does this in both its production and legal arms; the
            capture/ranker/rescue scripts never did)

Arms (fold's variant bank, same legal pass, exact scorer). The reject for the
base pool is fixed at "vote50 IoU > 0.5" as in §10b-d; rescues are the
clusters with vote50 IoU <= 0.5 and best-variant IoU > 0.5.

  prod            vote50, conf order                     (§10b prod)
  prod_post       vote50, conf order, post-carve reject  (~ oracle-prod)
  choice_conf     best variant on matched clusters, conf order
  choice_gt       same, GT-IoU order
  rescue_conf     vote50 + rescues (best variant), conf order
  rescue_gt       same, GT-IoU order
  full_conf       choice + rescue, conf order
  full_conf_post  choice + rescue, conf order, post-carve reject
  full_gt         choice + rescue, GT-IoU order
  full_gt_post    choice + rescue, GT-IoU order, post-carve reject (~ legal oracle)

Usage: python src/eda/candidate_bank_decompose.py --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament import IMG_H, IMG_W
from filament.pq import score
from filament.rle import decode_rle, encode_mask

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def legal_pass(chosen, post=None):
    """chosen: (stem, key, bank_row). Descending key per stem; each mask is
    carved against the union of earlier kept masks; dropped if < 64 px or,
    when `post` is given, if post(stem, carved_mask) is False."""
    preds = []
    by_stem = {}
    for stem, key, i in chosen:
        by_stem.setdefault(stem, []).append((key, i))
    for stem, lst in by_stem.items():
        lst.sort(key=lambda t: -t[0])
        taken = np.zeros((IMG_H, IMG_W), bool)
        k = 0
        for key, i in lst:
            mm = DEC(i) & ~taken
            if mm.sum() < 64:
                continue
            if post is not None and not post(stem, mm):
                continue
            taken |= mm
            k += 1
            preds.append((f"{stem}_{k}", encode_mask(mm.astype(np.uint8))))
    return pd.DataFrame(preds, columns=["filament_id", "segmentation_rle"])


def main():
    global DEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    vb = pd.read_parquet(OUT / f"variant_bank_fold{a.fold}.parquet")
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(vb["stem"]))]
    gt_df = gt[["filament_id", "segmentation_rle"]]
    rle_of = dict(zip(vb.index, vb["rle"]))
    DEC = lambda i: decode_rle(rle_of[i]).astype(bool)  # noqa: E731

    # GT layers per stem, decoded once, for the post-carve check
    gt_rle = {s: list(g["segmentation_rle"]) for s, g in gt.groupby("stem")}
    gt_cache = {}

    def gt_layers(stem):
        if stem not in gt_cache:
            gt_cache[stem] = [decode_rle(r).astype(bool) for r in gt_rle.get(stem, [])]
        return gt_cache[stem]

    def post_ok(stem, m):
        ma = m.sum()
        for g in gt_layers(stem):
            inter = (g & m).sum()
            if inter and inter / (ma + g.sum() - inter) > 0.5:
                return True
        return False

    vb["row"] = vb.index
    base = vb[vb["variant"] == "vote50"].set_index("cluster_id")
    non = vb[(vb["variant"] != "vote50") & vb["cluster_id"].isin(base.index)]
    best = non.loc[non.groupby("cluster_id")["iou"].idxmax()].set_index("cluster_id").reindex(base.index)
    # the oracle's choice per cluster: best over ALL variants incl. vote50
    use_best = best["iou"] > base["iou"]
    matched = base["iou"] > 0.5
    rescuable = (~matched) & (best["iou"] > 0.5)
    log(f"fold {a.fold}: {len(base)} clusters, {int(matched.sum())} matched, "
        f"{int(rescuable.sum())} rescuable, {len(gt)} GT rows")

    def rows(sel, choice, order):
        """(stem, key, bank_row) per selected cluster: the oracle-best variant
        when `choice` and it beats vote50, else vote50; key = production
        confidence or the mask's (pre-carve) GT IoU."""
        out = []
        for cid in base.index[sel]:
            if choice and use_best[cid]:
                i, iou = int(best.loc[cid, "row"]), float(best.loc[cid, "iou"])
            else:
                i, iou = int(base.loc[cid, "row"]), float(base.loc[cid, "iou"])
            key = float(base.loc[cid, "conf"]) if order == "conf" else iou
            out.append((base.loc[cid, "stem"], key, i))
        return out

    arms = {
        "prod":           (rows(matched, False, "conf"), None),
        "prod_post":      (rows(matched, False, "conf"), post_ok),
        "choice_conf":    (rows(matched, True, "conf"), None),
        "choice_gt":      (rows(matched, True, "gt"), None),
        "rescue_conf":    (rows(matched, False, "conf") + rows(rescuable, True, "conf"), None),
        "rescue_gt":      (rows(matched, False, "gt") + rows(rescuable, True, "gt"), None),
        "full_conf":      (rows(matched | rescuable, True, "conf"), None),
        "full_conf_post": (rows(matched | rescuable, True, "conf"), post_ok),
        "full_gt":        (rows(matched | rescuable, True, "gt"), None),
        "full_gt_post":   (rows(matched | rescuable, True, "gt"), post_ok),
    }
    res = {}
    log(f"\n  {'arm':<16}{'preds':>7}{'PQ':>9}{'SQ':>9}{'RQ':>9}{'FN':>6}{'FP':>6}{'Δ vs prod':>11}")
    for name, (ch, post) in arms.items():
        pdf = legal_pass(ch, post)
        r = score(gt_df, pdf)
        res[name] = dict(n=len(pdf), pq=r["pq"], sq=r["sq"], rq=r["rq"], fn=r["fn"], fp=r["fp"])
        log(f"  {name:<16}{len(pdf):>7}{r['pq']:>9.4f}{r['sq']:>9.4f}{r['rq']:>9.4f}{r['fn']:>6}{r['fp']:>6}"
            f"{r['pq']-res['prod']['pq']:>+11.4f}")

    p = {k: v["pq"] for k, v in res.items()}
    log("\n  ingredients, each as a realised PQ difference:")
    log(f"    choice alone (conf order)              {p['choice_conf']-p['prod']:>+8.4f}")
    log(f"    rescue alone (conf order)              {p['rescue_conf']-p['prod']:>+8.4f}")
    log(f"    choice + rescue (conf order)           {p['full_conf']-p['prod']:>+8.4f}")
    log(f"    + GT order instead of conf             {p['full_gt']-p['full_conf']:>+8.4f}")
    log(f"    + post-carve GT reject (on GT order)   {p['full_gt_post']-p['full_gt']:>+8.4f}")
    log(f"    post-carve GT reject on prod alone     {p['prod_post']-p['prod']:>+8.4f}")
    log(f"    post-carve GT reject on full_conf      {p['full_conf_post']-p['full_conf']:>+8.4f}")
    log(f"    full_gt_post vs prod (this framework's ceiling) {p['full_gt_post']-p['prod']:>+8.4f}")
    log(f"    reachable without GT at inference: choice/rescue with conf order, no post reject")
    json.dump(dict(fold=a.fold, res=res), open(OUT / f"candidate_bank_decompose_fold{a.fold}.json", "w"), indent=1)
    log(f"\nwrote {OUT/f'candidate_bank_decompose_fold{a.fold}.json'}")


if __name__ == "__main__":
    main()
