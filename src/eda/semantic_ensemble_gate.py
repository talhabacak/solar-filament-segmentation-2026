"""Talha's combine question, measured on maps that already exist.

Seven A100 arms of 2026-09-05 (wider decoder, 2048/B2, UPerNet, B5, SAAF,
GroupNorm, radial denoise) each fell below the production fold-0 model on the
instance gate. Each was judged as a REPLACEMENT. Talha's point: members that
lose alone can win combined -- and the ledger agrees in principle (recipe-
diverse ensemble, section 1 row 9: three recipes, pairwise corr 0.91-0.93,
+0.0054 pixel-Dice; the blob member at w=0.5: +0.006 PQ on five folds). The
seven arms differ by architecture and input, which is the "different mistakes"
kind of difference the seed/size ensembles lacked.

So: average the fold-0 probability maps and run the SAME instance gate
(gate_instance_quality: net rows reached at IoU>0.5 on both populations,
per-instance Dice, pixel-Dice) on every combination worth asking about:

  pair      base + one arm             does any single arm add when averaged?
  all       base + every arm           the full committee
  loo       all minus one arm          which member the committee cannot spare
  arms      every arm without base     is the production model itself needed?

and, first, the pairwise correlation of the maps, so a gain can be read as
"different mistakes averaged out" rather than taken on faith.

--fixed TAG:W keeps a fixed member at weight W outside the average (the
production OOF pool is 0.5*blob + 0.5*cldice, so the production-relevant
question is 0.5*blob + 0.5*mean(cldice, arms) against 0.5*blob + 0.5*cldice).

Usage:
  python src/eda/semantic_ensemble_gate.py --members 1536wdec4832 2048b2 upernet1536 \
      b5_1536 saaf1536 gn8_1536 rdn08 --out a100arms
  python src/eda/semantic_ensemble_gate.py --members ... --fixed blob1536:0.5 --out a100arms_blob
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
from filament.rle import decode_rle  # noqa: E402
from gate_instance_quality import cc, best_iou, THR_PROD, HI  # noqa: E402

SEM = ROOT / "outputs" / "semantic"
OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


class Member:
    def __init__(self, tag, fold):
        self.tag = tag
        self.z = np.load(SEM / f"fold{fold}_probs_{tag}.npz")

    def __call__(self, stem):
        a = np.asarray(self.z[stem])
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        return a.astype(np.float32)


def load_rows(fold):
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == fold, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    d = pd.read_parquet(OUT / "error_profile_fn_gt.parquet")
    d["matched"] = d["matched"].astype(bool)
    d["in_raw_pool"] = d["in_raw_pool"].astype(bool)
    d = d[d["stem"].isin(f0)].copy()
    d["grp"] = np.where(d["matched"], "matched",
                        np.where(d["in_raw_pool"], "in-pool", "never-proposed"))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt = gt.set_index("filament_id")["segmentation_rle"]
    return d, gt


GRPS = ("matched", "never-proposed", "in-pool")


def measure(get_prob, rows, gt, thrs=(THR_PROD,)):
    """One pass over the maps, the gate at every threshold in `thrs`. An
    ensemble mean is more conservative than any member where members disagree,
    so judging it at the single model's 0.3 confounds calibration with
    information; the sweep separates the two. Returns {thr: result}."""
    out = {t: [] for t in thrs}
    inter = {t: 0 for t in thrs}
    union = {t: 0 for t in thrs}
    for stem, g in rows.groupby("stem"):
        prob = get_prob(stem)
        if prob is None:
            continue
        masks = []
        fg = np.zeros(prob.shape, bool)
        for r in g.itertuples():
            m = decode_rle(gt.loc[r.filament_id]).astype(bool)
            if not m.any():
                continue
            fg |= m
            masks.append((r.grp, m, float((prob[m] >= HI).mean())))
        for t in thrs:
            lab, keep = cc(prob >= t)
            pred = np.isin(lab, keep) if keep else np.zeros(prob.shape, bool)
            for grp, m, cov in masks:
                iou = best_iou(m, lab, keep)
                out[t].append(dict(grp=grp, iou=iou, dice=2 * iou / (1 + iou), coverage=cov))
            inter[t] += int((pred & fg).sum())
            union[t] += int(pred.sum() + fg.sum())
    res = {}
    for t in thrs:
        df = pd.DataFrame(out[t])
        reach = {g: int((df[df.grp == g]["iou"] > 0.5).sum()) for g in GRPS}
        res[t] = dict(net=sum(reach.values()), reach=reach,
                      dice_all=float(df["dice"].mean()),
                      dice={g: float(df[df.grp == g]["dice"].mean()) for g in GRPS},
                      cov_np=float(df[df.grp == "never-proposed"]["coverage"].mean()),
                      pixel_dice=2 * inter[t] / max(union[t], 1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cldice")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--fixed", default=None,
                    help="TAG:W kept outside the average at weight W")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--combos", default="pair,all,loo,arms",
                    help="comma list of pair,all,loo,arms, or explicit sets as "
                         "name=tag+tag+... (base is spelled by its tag)")
    ap.add_argument("--thrs", default="0.3",
                    help="extraction thresholds to gate at (production 0.3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, gt = load_rows(a.fold)
    stems = sorted(rows["stem"].unique())
    base = Member(a.base, a.fold)
    mem = {t: Member(t, a.fold) for t in a.members}
    fixed, fw = None, 0.0
    if a.fixed:
        ft, w = a.fixed.split(":")
        fixed, fw = Member(ft, a.fold), float(w)
    log(f"fold {a.fold}: {len(stems)} stems, {len(rows)} GT rows; base {a.base}; "
        f"members {a.members}; fixed {a.fixed}")

    # 1. how different are the maps? Pearson over an 8-px lattice of every stem.
    names = [a.base] + a.members
    vecs = {n: [] for n in names}
    for s in stems:
        vecs[a.base].append(base(s)[::8, ::8].ravel())
        for t, m in mem.items():
            vecs[t].append(m(s)[::8, ::8].ravel())
    X = np.stack([np.concatenate(vecs[n]) for n in names])
    C = np.corrcoef(X)
    log("\npairwise correlation of probability maps (base first):")
    log("  " + " " * 14 + " ".join(f"{n[:8]:>8s}" for n in names))
    for i, n in enumerate(names):
        log(f"  {n:14s} " + " ".join(f"{C[i, j]:8.3f}" for j in range(len(names))))
    del X, vecs

    def make(members_list):
        def get(stem):
            p = np.mean([m(stem) for m in members_list], axis=0)
            if fixed is not None:
                p = fw * fixed(stem) + (1 - fw) * p
            return p
        return get

    combos = []
    kinds = a.combos.split(",")
    if "pair" in kinds:
        combos += [(f"base+{t}", [base, mem[t]]) for t in a.members]
    if "all" in kinds:
        combos += [("base+all", [base] + list(mem.values()))]
    if "loo" in kinds:
        combos += [(f"all-minus-{t}", [base] + [m for u, m in mem.items() if u != t])
                   for t in a.members]
    if "arms" in kinds:
        combos += [("arms-only", list(mem.values()))]
    for k in kinds:
        if "=" in k:                       # explicit set: name=tag+tag+...
            name, tags = k.split("=", 1)
            pool = {**mem, a.base: base}
            combos.append((name, [pool[t] for t in tags.split("+")]))

    thrs = tuple(float(x) for x in a.thrs.split(","))
    t0 = time.time()
    bb = measure(make([base]), rows, gt, thrs)
    fx = f" with fixed {a.fixed}" if a.fixed else ""
    for t in thrs:
        b = bb[t]
        log(f"\nbase {a.base}{fx} @thr {t:.2f}: net {b['net']} (matched {b['reach']['matched']}, "
            f"never-proposed {b['reach']['never-proposed']}, in-pool {b['reach']['in-pool']}), "
            f"Dice-all {b['dice_all']:.4f}, pixel-Dice {b['pixel_dice']:.4f}"
            f"  ({time.time()-t0:.0f}s)")
    # every delta is against the base AT THE PRODUCTION THRESHOLD: a combination
    # is only useful if it beats what production reaches today, whatever
    # threshold it needs to do so.
    b = bb[THR_PROD] if THR_PROD in bb else bb[thrs[0]]
    log(f"\n{'combination':26s} {'thr':>5s} {'net':>5s} {'dNet':>6s} {'dMat':>6s} {'dNP':>6s} "
        f"{'dPool':>6s} {'dDice':>8s} {'dPix':>8s}")
    res = {"base": {str(t): bb[t] for t in thrs},
           "corr": {n: {m: float(C[i, j]) for j, m in enumerate(names)}
                    for i, n in enumerate(names)},
           "combos": {}}
    for name, ms in combos:
        t0 = time.time()
        rr = measure(make(ms), rows, gt, thrs)
        res["combos"][name] = {str(t): rr[t] for t in thrs}
        for t in thrs:
            r = rr[t]
            log(f"{name:26s} {t:5.2f} {r['net']:5d} {r['net']-b['net']:+6d} "
                f"{r['reach']['matched']-b['reach']['matched']:+6d} "
                f"{r['reach']['never-proposed']-b['reach']['never-proposed']:+6d} "
                f"{r['reach']['in-pool']-b['reach']['in-pool']:+6d} "
                f"{r['dice_all']-b['dice_all']:+8.4f} {r['pixel_dice']-b['pixel_dice']:+8.4f}"
                f"  ({time.time()-t0:.0f}s)")
        json.dump(res, open(OUT / f"semantic_ensemble_gate_{a.out}.json", "w"), indent=1)
    log(f"\nwrote {OUT / f'semantic_ensemble_gate_{a.out}.json'}")


if __name__ == "__main__":
    main()
