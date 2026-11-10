#!/usr/bin/env bash
# SDA-2021 runs only after the GPU queue is done: it died twice from CUDA OOM
# while other jobs shared the 8 GB card.
set -Eeuo pipefail
cd /f/ConfusionAttack
RUN_ID=${1:?provide the exact queue run ID}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid run ID" >&2; exit 2; }
STATE_DIR="reports/closed_loop/queue_runs/$RUN_ID"
[[ -d "$STATE_DIR" ]] || { echo "queue run not found: $RUN_ID" >&2; exit 2; }
PY="E:/miniconda/envs/attack-verify/python.exe"
LOG="$STATE_DIR/sda2021.log"
while true; do
  state=$(cat "$STATE_DIR/status" 2>/dev/null || true)
  case "$state" in
    SUCCEEDED) break ;;
    FAILED|INTERRUPTED) echo "queue $RUN_ID ended with $state" >&2; exit 1 ;;
    RUNNING)
      pid=$(cat "$STATE_DIR/pid" 2>/dev/null || true)
      [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || { echo "queue process is not running: $RUN_ID" >&2; exit 1; }
      sleep 60 ;;
    *) echo "missing or invalid queue state for $RUN_ID" >&2; exit 1 ;;
  esac
done
status() { printf '%s\n' "$1" > "$STATE_DIR/sda2021.status.tmp"; mv "$STATE_DIR/sda2021.status.tmp" "$STATE_DIR/sda2021.status"; }
trap 'status FAILED' ERR
trap 'status INTERRUPTED; exit 130' INT TERM
mkdir "$STATE_DIR/sda2021.lock" || { echo "SDA-2021 already started for $RUN_ID" >&2; exit 1; }
status RUNNING
echo "[sda2021] queue $RUN_ID finished, starting SDA-2021" >> "$LOG"
KMP_DUPLICATE_LIB_OK=TRUE "$PY" scripts/rm_sda_full.py --seeds 0,1 \
  --num-train 100000 --epochs 8 --beam 10 >> "$LOG" 2>&1
echo "[sda2021] SDA-2021 DONE" >> "$LOG"
status SUCCEEDED
