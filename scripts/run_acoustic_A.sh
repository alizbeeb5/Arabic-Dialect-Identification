#!/usr/bin/env bash
# One GPU for extraction, one CPU thread for the 16,980-parameter linear head.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared Arabic-Dialect-Identification-data folder}"
PYTHON="${PYTHON:-python}"
TASK_OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/outputs/week2_acoustic_A}"
TASK_CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/cache/huggingface}"
TASK_RUNS_ROOT="${RUNS_ROOT:-$DATA_ROOT/runs/week2_acoustic_A}"
TASK_DEVICE="${DEVICE:-cuda}"
TASK_REVISION=c8da454c9f474bb075d3fe608940ee29e8ccfa43
TASK_AUDIO_ROOT="${AUDIO_ROOT:-$DATA_ROOT/audio/adi20_micro/$TASK_REVISION}"
TASK_MANIFEST_ROOT="${MANIFEST_ROOT:-src/data/manifests}"
case "$MODE" in
  smoke|overfit)
    TASK_CONFIG=configs/week2_acoustic_A_smoke.json
    TASK_TRAIN=data/manifests/train_pilot.jsonl
    TASK_DEV=data/manifests/dev_pilot.jsonl
    TASK_SEEDS=(11)
    ;;
  preflight|select|report)
    TASK_CONFIG=configs/week2_acoustic_A.json
    TASK_TRAIN="$TASK_MANIFEST_ROOT/train.jsonl"
    TASK_DEV="$TASK_MANIFEST_ROOT/dev.jsonl"
    TASK_SEEDS=(11)
    if [[ "$MODE" == report ]]; then TASK_SEEDS=(17 29 43); fi
    ;;
  *) echo "Usage: $0 {overfit|smoke|preflight|select|report}" >&2; exit 1 ;;
esac
mkdir -p "$TASK_RUNS_ROOT" "$TASK_OUTPUT_ROOT" "$TASK_CACHE_ROOT"
for TASK_SEED in "${TASK_SEEDS[@]}"; do
  TASK_RUN="$TASK_RUNS_ROOT/${MODE}-seed${TASK_SEED}-$(date -u +%Y%m%dT%H%M%S%N)"
  TASK_COMMAND=("$PYTHON" -u src/experiments/acoustic_classifier.py
    --config "$TASK_CONFIG" --seed "$TASK_SEED"
    --train-manifest "$TASK_TRAIN" --train-metadata "${TASK_TRAIN%.jsonl}.metadata.json"
    --dev-manifest "$TASK_DEV" --dev-metadata "${TASK_DEV%.jsonl}.metadata.json"
    --audio-root "$TASK_AUDIO_ROOT" --output-root "$TASK_OUTPUT_ROOT"
    --cache-root "$TASK_CACHE_ROOT" --device "$TASK_DEVICE" --output-dir "$TASK_RUN")
  if [[ "$MODE" == overfit ]]; then TASK_COMMAND+=(--overfit-check-only); fi
  if [[ "$MODE" == preflight ]]; then TASK_COMMAND+=(--preflight-only); fi
  echo "Run: $TASK_RUN"
  # Log next to the output directory: the Python runner must create a fresh directory.
  set +e
  "${TASK_COMMAND[@]}" 2>&1 | tee "$TASK_RUN.console.log"
  TASK_EXIT=${PIPESTATUS[0]}
  set -e
  echo "Exit code: $TASK_EXIT"
  # Exit 2 is an explicit partial result. Preserve it and stop for diagnosis.
  if [[ "$TASK_EXIT" != 0 ]]; then exit "$TASK_EXIT"; fi
done
