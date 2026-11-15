"""E213b -- where does the hf-FiLM gain land? Validation Dice by hf, control against FiLM.

The FiLM run beat its paired control on the year holdout (best val pixel-Dice 0.6578 against
0.6437, ahead in all fifteen of epochs 15-29). If the conditioning does what it was built for, the
gain should grow with the frame's hf: ledger 46 put the whole regime loss in the busy frames. If it
is flat across hf, the FiLM layers are acting as generic extra capacity or the pair is a seed draw.

Both best.pt checkpoints are run on the validation frames exactly as the trainer validates them
(CachedSemantic, train=False, per annotator entry, threshold 0.5), Dice is pooled per image and per
hf quartile of the training frames, and the per-image difference is set against hf.

Usage: python src/eda/film_regime_check.py [ctl_run] [film_run]   (CPU is fine: 63 frames)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import semantic_candidates as SC  # noqa: E402
from semantic_candidates import load_model  # noqa: E402
from train_semantic import CachedSemantic  # noqa: E402

# CPU unless asked: the local card belongs to another project, and torch.cuda.is_available() is
# not a safe test here (it returns True with CUDA_VISIBLE_DEVICES="" while device_count() is 0).
# load_model reads this module global at call time.
SC.DEVICE = "cuda" if "--gpu" in sys.argv else "cpu"
sys.argv = [a for a in sys.argv if a != "--gpu"]

OUT = ROOT / "outputs" / "eda"


def per_entry(run, va):
    m, size, ch = load_model(ROOT / "outputs/runs_semantic" / run / "best.pt")
    ds = CachedSemantic(va, size, train=False, channels=ch)
    dev = next(m.parameters()).device
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i][0], ds[i][1]
            p = torch.sigmoid(m(x.unsqueeze(0).to(dev))[:, :1])[0].cpu() > 0.5
            y = y > 0.5
            rows.append(((p & y).sum().item(), p.sum().item() + y.sum().item()))
            if i % 20 == 0:
                print(f"  {run}: {i}/{len(ds)}", flush=True)
    return np.array(rows, float)


def main():
    ctl, film = (sys.argv[1:3] if len(sys.argv) > 2 else ("sem_yr_ctl2", "sem_yr_film"))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data/splits/folds_year.parquet")
    fold_of = dict(zip(folds["file_name"], folds["fold"]))
    e = gt[["entry_id", "file_name"]].drop_duplicates()
    va = list(e.loc[e["file_name"].map(fold_of) == 0, ["entry_id", "file_name"]].itertuples(index=False, name=None))
    A, B = per_entry(ctl, va), per_entry(film, va)
    d = pd.DataFrame({"stem": [fn.replace(".jpeg", "") for _, fn in va],
                      "ia": A[:, 0], "sa": A[:, 1], "ib": B[:, 0], "sb": B[:, 1]})
    img = d.groupby("stem").sum()
    q = pd.read_parquet(OUT / "image_quality.parquet")
    qt = q[q.split == "train"]
    edges = qt["hf"].quantile([.25, .5, .75]).to_numpy()
    img = img.join(qt.set_index("stem")["hf"])
    img["hq"] = pd.cut(img["hf"], [-np.inf, *edges, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
    dice = lambda i, s: 2 * i.sum() / max(s.sum(), 1)
    tot_a, tot_b = dice(d.ia, d.sa), dice(d.ib, d.sb)
    print(f"\npooled val pixel-Dice: control {tot_a:.4f}  FiLM {tot_b:.4f}  ({tot_b - tot_a:+.4f})"
          "  -- must match the trainer's best val")
    res = {"control": tot_a, "film": tot_b, "by_quartile": {}}
    for hq, g in img.groupby("hq", observed=True):
        a, b = dice(g.ia, g.sa), dice(g.ib, g.sb)
        res["by_quartile"][str(hq)] = dict(n=len(g), control=a, film=b)
        print(f"  {hq}: {len(g):2d} frames  control {a:.4f}  FiLM {b:.4f}  ({b - a:+.4f})")
    img["diff"] = 2 * img.ib / img.sb.clip(lower=1) - 2 * img.ia / img.sa.clip(lower=1)
    r, p = spearmanr(img["hf"], img["diff"])
    res["spearman_hf_diff"] = [float(r), float(p)]
    print(f"  per-frame Dice difference against hf: Spearman {r:+.3f} (p {p:.3f}); frames improved "
          f"{int((img['diff'] > 0).sum())}/{len(img)}")
    json.dump(res, open(OUT / f"film_regime_check_{film}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
