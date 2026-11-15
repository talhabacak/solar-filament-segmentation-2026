#!/usr/bin/env bash
# Folds 1-4 of the wide-decoder semantic model, sequentially on one GPU.
#
# Same recipe as fold 0 (sem_b3_f0_1536_cldice_wdec), which passed the instance
# gate at 768; the 1536 arm is (40,24) at batch 2, because batch 1 corrupts BatchNorm (-53 net rows on the
# baseline alone), and (40,24) is the widest decoder that fits at batch 2 (12.1 GB).
# Sequential, never concurrent: a concurrent GPU job is what killed blob fold 2.
# After each fold the out-of-fold map is generated so the pool can be built
# without a second pass.
set -u
cd "$(dirname "$0")/.."
for k in 1 2 3 4; do
  echo "=== fold $k $(date +%H:%M) ==="
  python src/models/train_semantic.py --fold "$k" --imgsz 1536 --epochs 30 --cldice 0.3 --cache \
      --workers 2 --batch 2 --grad-ckpt --decoder-channels 256,128,64,40,24 \
      --name "sem_b3_f${k}_1536_cldice_wdec4024" || { echo "fold $k FAILED"; exit 1; }
  python src/models/semantic_candidates.py \
      --weights "outputs/runs_semantic/sem_b3_f${k}_1536_cldice_wdec4024/best.pt" \
      --fold "$k" --tag _1536wdec4024 || { echo "fold $k OOF FAILED"; exit 1; }
done
echo WDEC_FOLDS_DONE
