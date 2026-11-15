#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python src/eda/radial_denoise_profile.py
for r in 50 60 70 80; do
  echo "=== GATE r*=0.$r (baseline inside, denoised outside) ==="
  python src/eda/gate_instance_quality.py --base outputs/semantic/fold0_probs_cldice.npz --cand outputs/semantic/fold0_probs_dn_r$r.npz | sed -n '/1\. PER-INSTANCE/,/wrote/p'
done
echo RADIAL_DONE
