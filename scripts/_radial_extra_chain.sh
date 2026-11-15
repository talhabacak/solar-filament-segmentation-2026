#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python outputs/eda/_radial_extra.py
for r in 75 85; do
  echo "=== GATE r*=0.$r ==="
  python src/eda/gate_instance_quality.py --base outputs/semantic/fold0_probs_cldice.npz --cand outputs/semantic/fold0_probs_dn_r$r.npz | sed -n '/1\. PER-INSTANCE/,/wrote/p'
done
echo EXTRA_DONE
