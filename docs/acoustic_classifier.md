# Task 12: frozen-encoder acoustic classifier

Linear: [EEC-16 / W1-12](https://linear.app/eece490/issue/EEC-16/w1-12-train-and-evaluate-the-frozen-encoder-acoustic-classifier).
Purpose: the first learned acoustic model, and the matched classifier reference for the
verifier experiment ([verifier_experiment.md](verifier_experiment.md)). Results:
[../reports/acoustic_classifier_pilot_review.md](../reports/acoustic_classifier_pilot_review.md).

**Result role:** pilot reference on the frozen split `w1-07-pilot-v1-e61cd569ac7b`. The development
pilot is in-source ADI-20, not out-of-domain.

## Run it

Colab (T4 GPU): open `notebooks/W1_12_Acoustic_Classifier.ipynb` and run the cells in order.
`MODE = 'overfit_check'` extracts all 700 pilot features into the cache on Drive and runs the
overfit gate only; `MODE = 'full'` reuses the cache, trains the head, predicts the dev pilot and runs
the shared evaluator. The notebook defaults to fetching reviewed code by immutable `PINNED_COMMIT`.
Optional ZIP mode includes `.git`, so a run records the base commit and any uncommitted files.

Command line, in the acoustic environment (`environments/acoustic.txt`), from the repository root:

```bash
python src/experiments/acoustic_classifier.py \
  --audio-root "$DATA_ROOT/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43" \
  --output-root "$OUTPUT_ROOT" --cache-root /local/disk/hf-cache --device cuda \
  --output-dir "$OUTPUT_ROOT/runs/acoustic_classifier_w1_12/<new-run-id>"
```

This one command regenerates the reported pilot predictions. The train/dev manifests default
to the committed frozen pilot and the configuration is `configs/acoustic_classifier.json`.
The output directory must not exist. `--audio-root` is only needed for clips that are not cached
yet; with a complete cache the run needs no audio. Add `--overfit-check-only` to stop after the
gate. Keep the Hugging Face cache on local disk: Drive corrupted the 1.24 GB checkpoint before.
Pilot audio can be fetched with `src/data/fetch_pilot_audio.py` (see
[mfcc_svm_baseline.md](mfcc_svm_baseline.md)).

Exit codes: `0` completed with no clip failures, `2` completed with explicit clip failures (still
written and scored), `1` setup/contract error or failed overfit gate.

## Pipeline

1. **Features, once.** The pinned encoder (`Elyadata/AraBEST-RQ-300M-6k`, revision
   `dff453e2b5f72f76211a523b1050691967e5a59b`, `acoustic-features-v2`) is loaded frozen and
   fills one `FeatureCache` for both splits ([acoustic_features.md](acoustic_features.md)).
   Clips already cached under the same provenance are reused; nothing is recomputed per split.
   All 26 layers are cached (mean pooling, fp32, no crop) so layer experiments need no GPU;
   the head reads `final` (layer index 25, 848 dimensions).
2. **Overfit gate.** The head trains on the first 2 training clips of every dialect (40 clips,
   300 full-batch updates, learning rate 1e-3). The run stops unless training accuracy reaches
   100% and the loss falls to at most 5% of its starting value. A starting loss near
   log(20) = 2.996 is expected; flat loss points to broken features, misjoined labels or the optimiser.
3. **Pilot training.** One linear layer `z = W h + b` (20 × 848 + 20 = 16,980 parameters), Xavier
   uniform W and zero b, unsmoothed cross-entropy. AdamW (betas 0.9/0.999, eps 1e-8, weight decay
   0.01 on W only), gradient clipping at norm 1.0, batch 256 with the last partial batch kept,
   30 epochs, no scheduler, no early stopping, last epoch kept. These match the verifier
   specification's classifier arm at its highest learning rate.
4. **Outputs.** All 20 dev logits in canonical label order, the learning curve, the checkpoint
   and the shared evaluator's metrics.

The head runs on CPU with one thread and `torch.use_deterministic_algorithms(True)`.
Initialisation and shuffling use independent seeds derived from the run seed (490), so a rerun
from the same cache writes a byte-identical checkpoint and predictions.

### Extraction batch size

`encoder.batch_size` is 1: every clip is encoded alone, with no padding, exactly as the official
model processes a single clip. With batch 8 a T4 ran out of memory on the dev pilot: one
relative-position attention tensor for 8 clips padded to the 142 s clip needs 6.0 GiB. At batch 1
the 142 s clip still peaks at 12.15 GiB allocated, so longer clips (up to 212 s in the full dev
release) need a larger GPU or the cropping decision in `acoustic_features.md`. Batch size is not
part of the cache key; extract all clips of one experiment with one setting.

## Frozen encoder and feature consistency

- `frozen_summary` refuses an encoder with any trainable parameter or in training mode.
- A SHA256 over every encoder parameter and buffer is taken before and after extraction and
  must match. The head only receives cached NumPy arrays, so no gradient can reach the encoder.
- The run record stores the checkpoint revision and file hashes, the feature config hash, the
  provenance hash and a hash of the exact cache entries (clip ID + cache key) each split read.
  Identical values across runs show that train and dev features came from the same cache.
- Each uncached WAV is checked against the manifest `audio_sha256` before extraction. Integrity
  and decoding failures become `audio_error` records. Failed extraction batches are retried one
  clip at a time; clips that still fail become `inference_error` records. Successful clips are
  cached and reused on retry. Failed training clips are excluded from fitting; every dev ID stays
  in the evaluation denominator. Cache/provenance/write errors stop the run as setup errors.

## Leakage controls

- Learning rate, epochs and the last-epoch checkpoint are fixed in the config; there is no search.
- Development rows enter feature extraction with identity fields only. Dev labels are read for the
  per-epoch learning curve and by the shared evaluator; no training decision uses them. A test
  confirms that permuting dev labels leaves the checkpoint and predictions byte-identical.

## Outputs (in the run folder)

| File | Content |
|---|---|
| `predictions.jsonl` | one row per dev clip in manifest order; `scores` are 20 logits in canonical order (`score_type: logits`) |
| `evaluation/metrics.json`, `evaluation/confusion_matrix.csv` | shared evaluator output |
| `checkpoint.pt` | head weights, labels, layer, feature-provenance hash and config |
| `learning_curve.jsonl` | per epoch: train and dev loss/accuracy, mean batch loss, largest gradient norm before clipping |
| `overfit_check.json` | gate subset, thresholds, result, feature statistics and curve |
| `failures.jsonl`, `config.json` | clip failures (empty when none) and the config used |
| `run_record.json` | commit and dirty flag, code/config hashes, encoder and frozen evidence, cache identity, feature statistics, counts, timings, GPU memory, package versions, artifact hashes |

`model_revision` is `arabest-rq-frozen-linear-pilot:<first 16 hex of sha256(checkpoint.pt)>`.
Checkpoints (`*.pt`) are git-ignored; the reported checkpoint stays with the run folder on Drive
and its hash is in the committed run record.

## Checks

```bash
python -m unittest discover -s tests -p 'test_acoustic_classifier*.py' -v
```

The tests use a tiny random encoder on synthetic audio (no checkpoint or network). They cover the
end-to-end outputs and frozen evidence, byte-identical reruns from the cache without audio,
dev-label permutation, a failed overfit gate, explicit audio failures, config rejection of an
unfrozen encoder or dev-based selection, and the notebook's source, mode and zip handling.

The committed Colab results are historical evidence from the original source at
`cc0fe5e6efea9cded41ba820391d1bcda159b434`. Review fixes add clip-failure isolation and a
pinned notebook setup; no new full-audio/GPU run is claimed for those fixes.
