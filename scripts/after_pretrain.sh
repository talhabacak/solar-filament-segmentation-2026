#!/usr/bin/env bash
# Wait for the running GONG pretraining to finish, then run the augmentation study.
#
# Measured reason for serialising rather than sharing the card: with both jobs
# resident the GPU sat at 15.4 / 16.3 GB and BOTH slowed down -- the pretraining
# from 345 to 488 s/epoch (+41%) and the study from 45 to ~450 s/epoch. That is
# the WDDM host-RAM spill, and it makes the parallel plan 11.3 h against 4.7 h
# for the same work run in sequence. Parallelism is only free while the card has
# headroom.
#
# Waits on the LOG, not a PID. Two Windows reasons: `ps -W` reports Windows PIDs
# that bash's own `kill -0` does not accept, and a finished run here routinely
# logs "done." while its persistent_workers children keep the process alive. So
# the sentinel is the line the trainer writes when it is finished, and any
# lingering children are cleared before the study starts.
#
# Usage: bash scripts/after_pretrain.sh [run-dir]
set -u
cd "$(dirname "$0")/.."
RUN="${1:-outputs/runs_pretrain/pre_gong_b3_768_std50}"
LOG=outputs/eda/aug_chain.log
mkdir -p outputs/eda
echo "waiting on $RUN ($(date +%H:%M:%S))" >> "$LOG"

while true; do
  if [ -f "$RUN/train.log" ] && grep -q "^done\." "$RUN/train.log"; then break; fi
  sleep 120
done
echo "pretrain finished $(date +%H:%M:%S)" >> "$LOG"

# Release the VRAM the finished-but-hung workers are still holding.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*pretrain_gong*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >/dev/null 2>&1
sleep 30
echo "starting augmentation study $(date +%H:%M:%S)" >> "$LOG"
bash scripts/aug_study.sh
