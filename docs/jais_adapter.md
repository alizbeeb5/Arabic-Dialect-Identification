# Task 10: Jais batch inference

The adapter turns prepared audio into transcripts and canonical dialect labels.
The model receives waveforms and original sample rates. The processor uses an
audio-only prompt, left padding, automatic dialect detection and greedy decoding.
Gold labels and file/source metadata do not enter model inputs. Defaults are the
tested IFM/Jais-ASR revision `1323995fa189c479369a3929be0e95039eba5cba`, FP16,
two clips per batch and 128 new tokens. The processor performs model-specific
resampling; the adapter averages stereo channels before processing.

## Environment

Use a separate Jais environment with hardware-compatible PyTorch, then run:

```bash
python -m pip install -r environments/jais.txt
```

The uploaded notebook evidence records Torch 2.11.0+cu130, Transformers 5.16.1,
Accelerate 1.15.0, NumPy 2.1.3, SoundFile 0.14.0 and soxr 1.1.0 on a Tesla T4.
These are observed direct dependencies. The export does not contain a complete
environment lock or Python version. FP16 was used on the T4; a BF16 probe that
includes emulation does not establish native BF16 hardware support.

## Existing Colab model

From a checkout containing this module:

```python
from src.models.jais_adapter import JaisAdapter, run_cached_jais

adapter = JaisAdapter(model, processor)
records, counts = run_cached_jais(
    clips, audio_root, cache_root, adapter, batch_size=2,
)
```

`model` and `processor` must be the selected pinned checkpoint. This interface
reuses an already loaded model. The runner reads only `clip_id` and `audio_path`
from each manifest row; the model backend accepts IDs, waveforms and rates.

## Command line

Set `SMOKE_MANIFEST` and `SMOKE_METADATA` to the small smoke selection already
prepared for Task 10. Set the remaining roots to persistent team storage.

```bash
python src/models/jais_adapter.py \
  --manifest "$SMOKE_MANIFEST" --metadata "$SMOKE_METADATA" \
  --audio-root "$AUDIO_ROOT" --cache-root "$CACHE_ROOT/jais-adapter" \
  --output-dir "$OUTPUT_ROOT/jais-smoke-001" \
  --dtype float16 --batch-size 2 --max-new-tokens 128
```

Every row in the supplied manifest is processed. A 67,015-row training manifest
therefore requests 67,015 clips. Use the separate five-row smoke manifest for the
Task 10 check. Task 7 owns shared research split/pilot selection.

Outputs are `predictions.jsonl`, `raw_outputs.jsonl` and `run_record.json`.
Predictions use the shared hard-label contract; raw records also preserve the
generated suffix, token IDs, transcript, native tag, refusal and truncation.
Parser errors retain generated text/token evidence. Every valid requested clip
ID appears once, including audio/model failures, in manifest order. A generation
exception marks each clip in that batch as failed; retry with a smaller batch
when investigating an out-of-memory error. No class-score vectors are invented.

Truncation is recorded separately from the dialect decision: a recognized tag
can be a valid hard-label prediction even if the transcript is truncated.
Transcript consumers must inspect `refused` and `truncated`; an unavailable
truncation flag remains null. Accuracy/F1 can be computed by Task 8. The current
evaluator leaves C_avg unavailable, and these hard tags do not provide the
20-class score vectors required for a score-based comparison/export.

## Resume and provenance

Use the same cache root and a new output directory for each attempt. Receipts
include the clip ID, exact input-byte SHA256, immutable checkpoint, generation
defaults/options, prompt, preprocessing, mapping, software/hardware fingerprint,
batch size and adapter-source SHA256. Receipts are written through a temporary
file and rename; incomplete temporary files are not reused. Audio is hashed on
cache lookup, so a warm run still performs file I/O. Cache content hashes and
record contracts are checked before reuse. Invalid receipts reject the run.

Successful outputs, refusals and unknown tags can be reused. Read/decode,
generation and parsing failures remain visible and are retried. An interrupted
run can reuse completed batches. Notebook-v1 receipts are not migrated to the
packaged module: the packaged source has its own identity. A code/config change
invalidates affected receipts. Keep caches and generated runs outside Git.

The CLI records source/config/input hashes, versions, timestamps, device, peak
allocated GPU memory and cache/new-output counts. Its wall time includes model
loading and audio/cache processing and excludes final export. Interpret timing
with the cache counts; a cache-only run does not measure model throughput.
Exit codes: 0 complete, 2 complete with clip failures, 1 setup/contract/storage
error, 130 interrupted. A new output directory preserves earlier run records.

## Checks and remaining GPU check

```bash
python -m unittest discover -s tests -p 'test_jais_adapter.py' -v
python tests/check_jais_evidence.py /path/to/task10_smoke_bundle.zip
```

The offline suite uses fake inference to check isolation, resume, invalidation,
failure retention and the shared prediction contract. The evidence checker
recomputes the supplied five-clip metrics without loading audio or a model.
The original notebook was GPU-tested. Run `notebooks/W1_10_Jais_Smoke.ipynb`
against the packaged module before recording packaged-module GPU acceptance.
Keep EEC-14 In Progress until that check and another teammate's review pass.
