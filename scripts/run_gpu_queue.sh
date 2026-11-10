#!/usr/bin/env bash
# Serial GPU queue for the remaining closed-loop runs.
# Waits for the SDA-2021 job to release the GPU, then runs the rest one at a
# time (concurrent GPU jobs on this 8 GB card slow each other ~4x).
set -Eeuo pipefail
cd /f/ConfusionAttack
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid run ID" >&2; exit 2; }
STATE_DIR="reports/closed_loop/queue_runs/$RUN_ID"
mkdir -p "$(dirname "$STATE_DIR")"
mkdir "$STATE_DIR" || { echo "run ID already exists: $RUN_ID" >&2; exit 2; }
status() { printf '%s\n' "$1" > "$STATE_DIR/status.tmp"; mv "$STATE_DIR/status.tmp" "$STATE_DIR/status"; }
trap 'status FAILED' ERR
trap 'status INTERRUPTED; exit 130' INT TERM
printf '%s\n' "$$" > "$STATE_DIR/pid"
status RUNNING
printf 'queue run ID: %s\n' "$RUN_ID"
PY="E:/miniconda/envs/attack-verify/python.exe"
LOG="$STATE_DIR/queue.log"

wait_gpu_free() {
  # wait until no other job holds GPU memory (the SDA-2021 run releases it)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    printf '[queue] %s GPU memory used: %s MiB\n' "$(date -u +%FT%TZ)" "$used" >> "$LOG"
    if [ "${used:-99999}" -lt 3000 ]; then break; fi
    sleep 60
  done
}

echo "[queue] waiting for GPU to free" >> "$LOG"
wait_gpu_free
echo "[queue] GPU free, starting queue" >> "$LOG"

echo "[queue] 1/3 ISA astate 5e4 (plaintext HiddenState probe)" >> "$LOG"
KMP_DUPLICATE_LIB_OK=TRUE "$PY" scripts/rm_isa_full.py --steps 50000 \
  --profile probe --methods "astate:2,8" --prompts 0,1 >> "$LOG" 2>&1

echo "[queue] 2/3 SDA-2023 (1:1 + homophonic)" >> "$LOG"
KMP_DUPLICATE_LIB_OK=TRUE "$PY" scripts/rm_sda2023_full.py --seeds 0 \
  --num-train 60000 --epochs 6 --beam 10 >> "$LOG" 2>&1

echo "[queue] 3/3 VMA OSNIP sorted-L1 full vocab" >> "$LOG"
KMP_DUPLICATE_LIB_OK=TRUE "$PY" scripts/rm_vma_full.py --batch 4096 \
  --distance sorted_l1 --out vma_osnip_full.jsonl --tag vma_osnip \
  --layers 4,8,12 --prompts 0,1 >> "$LOG" 2>&1

echo "[queue] QUEUE DONE" >> "$LOG"
status SUCCEEDED
