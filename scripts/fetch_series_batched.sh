#!/usr/bin/env bash
# Batched fetch for any series (the per-process stall that hit HMI hits AIA too).
# Usage: bash scripts/fetch_series_batched.sh aia304 [max_frames]
cd "$(dirname "$0")/.."
SER=${1:-aia304}; MAX=${2:-0}; BATCH=${BATCH:-12}; DEADLINE=${DEADLINE:-420}
PFX=$(python -c "import sys; sys.path.insert(0,'src/data'); from fetch_magnetograms import SERIES; print(SERIES['$SER']['prefix'])")
while :; do
  have=$(ls outputs/magnetograms/${PFX}_*.npy 2>/dev/null | wc -l)
  left=$(python - <<PY
from pathlib import Path
R = Path("data/MAGFiLO_1.0_Kaggle_2026")
stems = [p.stem for d in ("train/train_images", "test/test_images") for p in (R / d).glob("*.jpeg")]
print(sum(1 for s in stems if not (Path("outputs/magnetograms") / f"${PFX}_{s}.npy").exists()))
PY
)
  echo "$(date '+%F %T') $SER: $have fetched, $left left"
  [ "$left" -le 3 ] && break
  [ "$MAX" -gt 0 ] && [ "$have" -ge "$MAX" ] && break
  timeout "$DEADLINE" python src/data/fetch_magnetograms.py --series "$SER" --workers 2 --limit "$BATCH" 2>&1 | grep -E " ok | no_record|failed" | tail -n 3
  sleep 3
done
echo "FETCH_${SER}_DONE"
