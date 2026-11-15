#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4; do
  echo "=== fold $k $(date +%H:%M) ==="
  python src/eda/candidate_bank_oracle.py --fold "$k" 2>&1 | grep -E "variant bank|^  prod|^  cluster|^  legal|LEGAL oracle|gate \(E094\)|images \(" 
done
echo ORACLE_FOLDS_DONE
