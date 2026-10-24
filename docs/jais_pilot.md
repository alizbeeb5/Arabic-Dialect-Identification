# Task 13: Jais on the frozen pilot

Linear: EEC-17 / W1-13. Run inference, not training. Use
`notebooks/W1_13_Jais_Pilot.ipynb` in a fresh Colab GPU runtime. First run setup,
input checks and the 20-clip smoke. Inspect its summary before running 500 train
and 200 dev clips. The notebook uses the packaged adapter's existing provenance,
cache and failure handling; adapter source is unchanged by this task.

## Inputs and outputs

Use committed `data/manifests/train_pilot.jsonl` (500) and `dev_pilot.jsonl` (200),
with their metadata. They reproduce `w1-07-pilot-v1-e61cd569ac7b`. Do not feed the
67,015-row full training manifest to this pilot. Gold labels only enter evaluation
and the separate target export; the adapter prompt remains audio-only.

The smoke is one clip per dialect, chosen by minimum clip ID within the frozen
training pilot. It is an engineering check, not a benchmark. `prepare-smoke`
records its parent manifest hash and writes adapter-compatible metadata.

Each adapter run preserves `predictions.jsonl`, `raw_outputs.jsonl` and
`run_record.json`. New output directories are mandatory on retries; preserve the
same cache/config to reuse completed work. Notebook-era caches have another
source identity and are not automatically migrated. FP16, batch size 2 and 128
new tokens retain Task 10 defaults. Inspect truncation in the smoke; any changed
generation limit must be recorded and used consistently for train and dev.

The standard-library export command validates input/run hashes, checkpoint,
expected IDs, raw/standard prediction agreement and counts before producing:

- `transcripts.jsonl`: ID, parsed text, text status and quality flags; no dialect
  prediction, gold label, source name, audio path or raw generation suffix.
- `targets.jsonl`: ID and gold target only, joined separately by Task 14.
- `diagnostics.jsonl`: native/canonical dialect, prediction failure and text flags.
- `direct_dialect_metrics.json`: the shared evaluator's metrics, including failures.
- `summary.json`: status counts, resource record, provenance and artifact hashes.

There is exactly one transcript row per requested ID, including unusable text.
Unknown dialect tags can still yield valid text. Refusals, empty text, errors,
truncation, unknown truncation flags and leftover generated tag markers are
explicitly distinguished. Default text usability requires a nonempty,
nonrefused, untruncated parsed transcript with no control/tag markers. This rule
is conservative; inspect unknown flags rather than silently treating them as
clean. The exporter does not append targets or strip ordinary spoken words
mentioning a dialect. It flags bracketed Jais dialect codes and `<|` markers;
this is a marker check, not proof that every possible model artifact is detected.

Task 14 must document how it handles unavailable/partial training text and keep
the full 200-clip dev denominator. Do not run TF-IDF on raw outputs, diagnostics
or targets. Do not silently remove rows where `usable_for_text_model` is false.

## CLI equivalents

From repository root; paths refer to actual adapter artifacts:

```bash
python src/experiments/jais_pilot.py prepare-smoke \
  --manifest data/manifests/train_pilot.jsonl \
  --metadata data/manifests/train_pilot.metadata.json \
  --output-dir "$SMOKE_INPUT_DIR"

python src/models/jais_adapter.py \
  --manifest "$SMOKE_INPUT_DIR/smoke.jsonl" \
  --metadata "$SMOKE_INPUT_DIR/smoke.metadata.json" \
  --audio-root "$AUDIO_ROOT" --cache-root "$JAIS_CACHE" \
  --model-cache "$HF_CACHE" --output-dir "$SMOKE_RUN_DIR" \
  --dtype float16 --batch-size 2 --max-new-tokens 128

python src/experiments/jais_pilot.py export \
  --manifest "$SMOKE_INPUT_DIR/smoke.jsonl" \
  --metadata "$SMOKE_INPUT_DIR/smoke.metadata.json" \
  --split train --run-dir "$SMOKE_RUN_DIR" --output-dir "$SMOKE_EXPORT_DIR"
```

For full pilot inference use the committed train/dev pilot and matching sidecar.
For export pass `--split train` or `--split dev` accordingly. Adapter exit 2 means
outputs exist with explicit clip failures: export and inspect them; it is not a
successful zero-failure run. Exit 1 is a setup/contract error. The notebook runs
the shared evaluator CLI too, producing its standard metrics and confusion CSV.

## Measurement and acceptance

Wall time includes model loading and I/O. Cache-only runs do not measure model
throughput. The summary reports cold end-to-end throughput only when all clips
succeed with zero cache hits. GPU-only generation time is not measured by the
existing adapter and is explicitly null; do not relabel wall time as GPU time.
Peak allocated GPU memory comes from the adapter record. Keep cold-run records,
not just a later cached rerun. The model/cache/download files remain outside Git.

Before calling the task complete, obtain real GPU results, inspect failures and
truncation, confirm 500/200 expected IDs, retain the full-dev evaluation, and have
a different teammate review. C_avg remains unavailable for these hard-label-only
Jais outputs. Task 12 later must use these same 200 dev IDs for comparison.

Offline checks (not GPU validation):

```bash
python -m unittest discover -s tests -p 'test_jais_pilot.py' -v
python -m unittest discover -s tests -p 'test_jais_adapter.py' -v
```
