"""Label a fused out-of-fold candidate pool for selector training.

The selector learns from a pool in which every candidate carries its target: whether it matches
an annotator copy (`y`) and the IoU it scores summed over the copies it matches (`iou_sum`). Each
candidate is first shrunk exactly as the submission path shrinks it (`mask_shrink.shrink_frac`
at 0.05, which removes the one-pixel boundary ring), and the area features are recomputed on
the shrunk mask, so the selector is fitted on the object it will judge at test time.

This is the step that produced outputs/eda/fused_pool_all_blob5_t020_labelled.parquet from
outputs/selector/fused_pool_all_blob5_t020.parquet (written by build_fused_pool.py). It needs the
ground-truth cache outputs/gt_cache/gt_full.parquet (src/eda/build_gt_cache.py).

Usage: python src/models/label_pool.py fused_pool_all_blob5_t020
Output: outputs/eda/<name>_labelled.parquet
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from two_stage_selector import label_pool  # noqa: E402

SHRINK = 0.05          # the submission's shrink setting; see mask_shrink.shrink_frac


def main():
    name = sys.argv[1]
    out = ROOT / "outputs" / "eda" / f"{name}_labelled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cand, n_gt, _ = label_pool(name + ".parquet", SHRINK, False)
    cand.to_parquet(out, index=False)
    print(f"{name}: {len(cand)} candidates, {int(cand['y'].sum())} positives, "
          f"{n_gt} ground-truth copies ({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
