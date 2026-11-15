"""Would focal loss up-weight signal, or annotator idiosyncrasy?

Focal multiplies each pixel's loss by (1 - p_t)^gamma, which is monotonically
DECREASING in the predicted probability of the true class. Its weight is maximal
where the model is most confidently wrong -- not, as I first said, where the
model is uncertain. The distinction decides the experiment, because we now know
what the confidently-wrong foreground pixels belong to.

The 282 zero-overlap ground-truth rows carry a median in-mask maximum probability
of 0.004, and their annotator agreement is the inverse of the rows we match: only
3.2% sit at three-of-three consensus against 28.3% of matched rows, while 23.8%
sit on entries where the annotators actively disagreed against 4.6%. If those are
the pixels focal would chase, it is being asked to learn one annotator's opinion
against the others', and the metric charges for that: a prediction only one of
three annotators drew scores one true positive and two false positives.

So the precondition is not "are there hard pixels" -- there always are. It is:

    among ground-truth foreground pixels, does annotator agreement FALL as the
    predicted probability falls?

If it does, focal's weight lands on the least corroborated pixels and the term
is contraindicated regardless of tuning. If agreement is flat across the
probability range, focal is re-weighting hard-but-real structure and is worth a
run.

The measurement is over pixels rather than instances, because that is what the
loss sees. Agreement is per pixel: how many of that image's annotators marked it
foreground, divided by how many annotated the image.

Usage: python src/eda/focal_precondition.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.rle import decode_rle

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
PROBS = "fold0_probs_cldice.npz"
GAMMA = 2.0
BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01]


def log(*a):
    print(*a, flush=True)


def main():
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    f0 = set(folds.loc[folds["fold"] == 0, "file_name"]
             .str.replace(".jpeg", "", regex=False))
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(f0)]
    z = np.load(SEM / PROBS)
    n_ann = gt.groupby("stem")["entry_id"].nunique()
    log(f"fold 0: {len(gt)} ground-truth rows over {gt['stem'].nunique()} images")
    log(f"annotators per image: " +
        ", ".join(f"{k} -> {v} images" for k, v in
                  n_ann.value_counts().sort_index().items()))

    # per-pixel agreement: how many annotators of this image marked it foreground
    prob_all, agree_all = [], []
    t0 = time.time()
    for stem, g in gt.groupby("stem"):
        if stem not in z:
            continue
        N = int(n_ann[stem])
        votes = None
        for eid, ge in g.groupby("entry_id"):
            m = np.zeros((2048, 2048), bool)
            for r in ge.itertuples():
                m |= decode_rle(r.segmentation_rle).astype(bool)
            votes = m.astype(np.uint8) if votes is None else votes + m
        fg = votes > 0
        if not fg.any():
            continue
        prob = np.asarray(z[stem]).astype(np.float32) / 255.0
        prob_all.append(prob[fg])
        agree_all.append(votes[fg].astype(np.float32) / N)
    p = np.concatenate(prob_all)
    a = np.concatenate(agree_all)
    log(f"\n{len(p):,} ground-truth foreground pixels ({time.time()-t0:.0f}s)")

    b = pd.cut(p, BINS, right=False)
    df = pd.DataFrame({"p": p, "agree": a, "bin": b})
    t = df.groupby("bin", observed=True).agg(
        pixels=("p", "size"), mean_agreement=("agree", "mean"),
        frac_full=("agree", lambda s: float((s >= 0.999).mean())))
    t["share_of_fg"] = 100 * t["pixels"] / len(p)
    t["focal_weight"] = [(1 - min(max(i.left, 0.0), 1.0)) ** GAMMA for i in t.index]
    t["share_of_focal_loss"] = 100 * (t["pixels"] * t["focal_weight"]
                                      / (t["pixels"] * t["focal_weight"]).sum())
    log(f"\nagreement and focal weight by predicted probability (gamma={GAMMA})")
    log(t.to_string(float_format=lambda v: f"{v:.3f}"))

    lo = df[df["p"] < 0.1]
    hi = df[df["p"] >= 0.6]
    log(f"\n  pixels the model calls background (<0.1): mean agreement "
        f"{lo['agree'].mean():.3f}, {100*(lo['agree'] >= 0.999).mean():.1f}% "
        f"unanimous")
    log(f"  pixels the model calls foreground (>=0.6): mean agreement "
        f"{hi['agree'].mean():.3f}, {100*(hi['agree'] >= 0.999).mean():.1f}% "
        f"unanimous")
    log(f"  focal at gamma {GAMMA} sends "
        f"{t.loc[t.index[:3], 'share_of_focal_loss'].sum():.1f}% of its "
        f"foreground loss to the <0.1 band, which holds "
        f"{t.loc[t.index[:3], 'share_of_fg'].sum():.1f}% of the pixels")

    log("\nreading: if agreement falls with probability, focal's weight is landing")
    log("on the least corroborated pixels. The metric scores a prediction against")
    log("EVERY annotator separately, so learning a minority opinion buys one true")
    log("positive and pays for the rest as false positives.")

    json.dump(dict(n_fg_pixels=int(len(p)),
                   agreement_low_band=float(lo["agree"].mean()),
                   agreement_high_band=float(hi["agree"].mean()),
                   unanimous_low=float((lo["agree"] >= 0.999).mean()),
                   unanimous_high=float((hi["agree"] >= 0.999).mean())),
              open(OUT / "focal_precondition.json", "w"), indent=1)
    log(f"\nwrote {OUT/'focal_precondition.json'}")


if __name__ == "__main__":
    main()
