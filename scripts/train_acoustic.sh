#!/usr/bin/env bash
# Shared interface: one GPU per process, no SLURM or cluster-specific paths.
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHITECTURE="${1:-all}"
STAGE="${2:-smoke}"
case "$ARCHITECTURE" in A|B|C) ARMS=("$ARCHITECTURE");; all) ARMS=(A B C);; *) echo 'Architecture must be A, B, C or all' >&2; exit 1;; esac
case "$STAGE" in smoke|preflight|select|report) ;; *) echo 'Stage must be smoke, preflight, select or report' >&2; exit 1;; esac
: "${DATA_ROOT:?Set DATA_ROOT to the prepared data folder}"
: "${RUNS_ROOT:?Set RUNS_ROOT to retained output storage}"
PYTHON="${PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
mkdir -p "$RUNS_ROOT/logs"
for ARM in "${ARMS[@]}"; do
  COMMAND=("$PYTHON" -u -m src.experiments.acoustic_abc train --architecture "$ARM" --stage "$STAGE"
    --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --device "${DEVICE:-cuda}")
  if [[ -n "${AUDIO_ROOT:-}" ]]; then COMMAND+=(--audio-root "$AUDIO_ROOT"); fi
  if [[ -n "${CACHE_ROOT:-}" ]]; then COMMAND+=(--cache-root "$CACHE_ROOT"); fi
  if [[ -n "${OUTPUT_ROOT:-}" ]]; then COMMAND+=(--output-root "$OUTPUT_ROOT"); fi
  LOG="$RUNS_ROOT/logs/${ARM}-${STAGE}-$(date -u +%Y%m%dT%H%M%S%N).log"
  echo "Architecture $ARM, stage $STAGE; log: $LOG"
  "${COMMAND[@]}" 2>&1 | tee "$LOG"
done
