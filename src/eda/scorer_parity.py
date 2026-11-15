"""Appendix C: verify our PQ implementation against the organizers' own scorer.

Every local number in the report comes from `src/filament/pq.py`. That is only
worth anything if it agrees with the scorer the competition actually runs, so
this re-runs the check as a reproducible artifact rather than a claim: the
organizers published their exact implementation in the Self-Evaluation
notebook, extracted here as `outputs/research/official_pq_scorer_extracted.py`,
and both are run on the same inputs.

Cases are chosen to exercise the semantics that shaped our design, not just the
happy path:
  * a real fold's predictions (the production case)
  * predictions that are perfect copies of the ground truth (SQ must be 1.0)
  * an empty prediction set (every GT becomes FN)
  * a single prediction against a multi-annotator image (the same mask scored
    against each copy independently)
  * masks just above and just below the IoU 0.5 boundary (the threshold is
    strict, so 0.5 exactly must NOT match)

Usage: python src/eda/scorer_parity.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import score as our_score
from filament.rle import encode_mask

OFFICIAL = ROOT / "outputs" / "research" / "official_pq_scorer_extracted.py"


def log(*a):
    print(*a, flush=True)


def load_official():
    """The published scorer is a two-step composition, not one entry point:
    get_overlap_df builds a per-annotator-image table of IoU matrices, and
    get_pq_score reduces it. Compose them so the comparison runs the official
    code path exactly as the notebook does."""
    spec = importlib.util.spec_from_file_location("official", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def scorer(gt_df, pred_df):
        return mod.get_pq_score(mod.get_overlap_df(gt_df, pred_df))

    return scorer, "get_overlap_df + get_pq_score"


def rect(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def df_from(pairs):
    return pd.DataFrame(pairs, columns=["filament_id", "segmentation_rle"])


def main():
    if not OFFICIAL.exists():
        raise SystemExit(f"missing {OFFICIAL}")
    official, entry = load_official()
    log(f"official scorer: {OFFICIAL.name}::{entry}\n")

    cases = []

    # synthetic cases first -- they pin the semantics
    gt1 = df_from([("040301-img_1", encode_mask(rect(100, 200, 100, 300)))])
    cases.append(("perfect copy", gt1, gt1.rename(
        columns={"filament_id": "filament_id"}).assign(
        filament_id=["img_1"])))
    cases.append(("empty prediction", gt1, df_from([])))

    # The IoU 0.5 boundary. Matching is strictly greater than 0.5, so a
    # prediction contained in the GT at exactly half its area must NOT match
    # (a first version of this suite mislabelled two cases at IoU 0.665 and
    # 0.855 as "below" and "above", testing nothing at the boundary).
    g = rect(100, 200, 100, 300)                       # 20,000 px
    boundary = {
        "IoU 0.4999 (must not match)": rect(100, 200, 100, 200 - 1),   # 9,900
        "IoU 0.5000 exactly (strict >, must not match)": rect(100, 200, 100, 200),
        "IoU 0.5001 (must match)": rect(100, 200, 100, 201),           # 10,100
    }
    for label, pm in boundary.items():
        inter = int((g & pm).sum())
        iou = inter / (int(g.sum()) + int(pm.sum()) - inter)
        cases.append((f"{label} [IoU={iou:.4f}]",
                      df_from([("040301-img_1", encode_mask(g))]),
                      df_from([("img_1", encode_mask(pm))])))

    # multi-annotator: one prediction scored against three copies
    gt3 = df_from([(f"04030{k}-img_1", encode_mask(rect(100, 200, 100, 300 + 10 * k)))
                   for k in (1, 2, 3)])
    cases.append(("one prediction, three annotator copies", gt3,
                  df_from([("img_1", encode_mask(rect(100, 200, 100, 310)))])))

    # and a real fold
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    files = sorted(folds.loc[folds["fold"] == 0, "file_name"].unique())[:40]
    gsub = gt[gt["file_name"].isin(files)][["filament_id", "segmentation_rle"]]
    pool = pd.read_parquet(ROOT / "outputs" / "selector" / "fused_pool_all.parquet")
    stems = {f.rsplit(".", 1)[0] for f in files}
    psub = pool[pool["stem"].isin(stems)].head(400)[["cand_id", "rle"]].rename(
        columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    cases.append(("real fold-0 subset (40 images, 400 candidates)", gsub, psub))

    log(f"{'case':<46s} {'ours':>9s} {'official':>9s} {'delta':>10s}")
    worst = 0.0
    for label, g_df, p_df in cases:
        a = our_score(g_df, p_df)
        b = official(g_df, p_df)
        pq_a = float(a["pq"] if isinstance(a, dict) else a)
        pq_b = float(b["pq"] if isinstance(b, dict) else b)
        d = abs(pq_a - pq_b)
        worst = max(worst, d)
        log(f"{label:<46s} {pq_a:9.6f} {pq_b:9.6f} {d:10.2e}")

    log(f"\nlargest absolute difference across cases: {worst:.2e}")
    log("PARITY OK" if worst < 1e-9 else "MISMATCH -- investigate before trusting any local number")


if __name__ == "__main__":
    main()
