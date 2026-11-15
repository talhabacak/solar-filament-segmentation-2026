#!/usr/bin/env bash
# Four runs of the fetcher have gone silent after about twenty files each, with no
# error and no timeout firing: per-call sessions, a pooled session, no keep-alive and
# short read timeouts all failed the same way. Whatever it is, it is per-PROCESS and
# it resets when the process does. So: run the fetcher in short batches, each in its
# own process with a hard deadline, and loop until nothing is left. A hung batch
# costs its deadline, not the run.
cd "$(dirname "$0")/.."
BATCH=${BATCH:-12}
DEADLINE=${DEADLINE:-420}
while :; do
  left=$(python - <<'PY'
from pathlib import Path
R = Path("data/MAGFiLO_1.0_Kaggle_2026")
stems = [p.stem for d in ("train/train_images", "test/test_images") for p in (R / d).glob("*.jpeg")]
print(sum(1 for s in stems if not (Path("outputs/magnetograms") / f"mag_{s}.npy").exists()))
PY
)
  # frames JSOC has no good-quality record for stay missing by design (they get the
  # neutral channels and flag 0); stop when only those are left
  norec=$(python -c "import pandas as pd,pathlib;p=pathlib.Path('outputs/magnetograms/index.parquet');print(int((pd.read_parquet(p)['status']=='no_record').sum()) if p.exists() else 0)")
  echo "$(date '+%F %T') $left left ($norec without a record)"
  [ "$left" -le "$norec" ] && break
  timeout "$DEADLINE" python src/data/fetch_magnetograms.py --workers 2 --limit "$BATCH" 2>&1 | grep -E " ok | no_record|failed|index:" | tail -n "$BATCH"
  sleep 3
done
echo FETCH_ALL_DONE
