#!/usr/bin/env bash
# Clean-environment rehearsal, step 1 of 2 (GPU): rebuild the champion's submission from the
# CHECKPOINTS, in today's environment, and keep every artefact for comparison.
#
# Why this has to be done before the deadline: the champion CSV (sub033) was built on
# 2026-09-02..07 with sklearn 1.6.1; the box now runs sklearn 1.7.2 (the selector bundles
# unpickle with an InconsistentVersionWarning), torch 2.11, cv2 5.0, numpy 2.3. And the test
# probability map is already known NOT to be bit-reproducible: a 2026-09-05 recomputation of the
# same recipe differs from the 2026-09-02 original by up to 3/255 per pixel. What matters is how
# far that reaches into the submitted RLEs, which only an end-to-end rebuild can say.
#
# Recipe of sub033 (ledger 15 section 19c, board rank 1):
#   --semantic recipe_blob5_50   12 members: 5x clDice 1/30, 2x heavy 1/6, 5x blob 0.1
#   --sem-thr 0.2  --conf 0.40  --p-min 0.60  --selector selector_v5_shrunk_t020pox (42 feats)
# The original map is moved aside (never deleted) so the members are recomputed from disk.
set -u
cd "$(dirname "$0")/../.."
SEM=outputs/semantic
ORIG=$SEM/test_probs_recipe_blob5_50.npz
KEEP=$SEM/test_probs_recipe_blob5_50_orig20260902.npz
md5sum "$ORIG" > outputs/eda/_rehearsal_orig.md5
[ -f "$KEEP" ] || cp "$ORIG" "$KEEP"
mv "$ORIG" "$SEM/_tmp_hidden_orig.npz"
restore() { [ -f "$SEM/_tmp_hidden_orig.npz" ] && mv -f "$SEM/_tmp_hidden_orig.npz" "$ORIG"; }
trap 'restore' EXIT
python src/models/submit_v2.py --out outputs/submissions/sub033_rehearsal_gpu.csv \
  --semantic recipe_blob5_50 --sem-thr 0.2 --conf 0.40 --p-min 0.60 \
  --selector outputs/selector/selector_v5_shrunk_t020pox.joblib
rc=$?
echo "submit rc=$rc"
# the rebuild wrote a FRESH map at the canonical path; keep it under its own name
[ -f "$ORIG" ] && mv "$ORIG" "$SEM/test_probs_recipe_blob5_50_rebuild$(date +%Y%m%d).npz"
restore; trap - EXIT
md5sum -c outputs/eda/_rehearsal_orig.md5 && echo "original map restored intact"
ls -la $SEM/test_probs_recipe_blob5_50*.npz
echo REHEARSAL_GPU_DONE
