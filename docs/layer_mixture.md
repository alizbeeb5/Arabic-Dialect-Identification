# Idea 1 pilot: layer sweep and learned layer mixture

Code: [src/experiments/layer_mixture.py](../src/experiments/layer_mixture.py).
Configuration: [configs/layer_mixture.json](../configs/layer_mixture.json).
Checks: [tests/test_layer_mixture.py](../tests/test_layer_mixture.py).

**Result role:** in-source ADI-20 pilot on the frozen split `w1-07-pilot-v1-e61cd569ac7b`
(500 train, 200 dev clips). It decides whether reading other encoder layers is worth pursuing.
It is not an out-of-domain or benchmark result. On 200 dev clips one clip is 0.5 points, so
differences under about 7 points are noise.

## Question

The W1-12 head reads only the final layer (index 25), averaged over time. Do other layers of
the frozen AraBEST-RQ encoder, or a learned mix of layers, separate dialects better?

## What it reads

Only the feature cache that `acoustic_classifier.py` already filled: 26 layers × 848 dimensions,
masked-mean pooled, for all 700 pilot clips. No encoder, GPU or audio is needed.
The script rebuilds the pinned provenance without loading the checkpoint. `FeatureCache` then
checks it against the stored `config.json` and every cache key, so a cache built from another
checkpoint, feature config or code version is refused. Any manifest clip that is missing, or
was cached from different audio, stops the run with exit code 1.

The W1-12 Colab run wrote this cache to
`MyDrive/Arabic-Dialect-Identification-data/output/features/arabest-rq-300m-6k/dff453e2b5f7/e4b3c287854e40f9/`
(about 62 MB). Pass the folder that contains `features/` as `--output-root`.

## Arms

Every head is the W1-12 20-way linear layer with the same AdamW settings, trained on CPU.
All arms except `b0_final_raw` z-score each layer's dimensions, with statistics from the
training rows of that fit only.

| Arm | Input |
|---|---|
| `b0_final` | final layer; the W1-12 head with a chosen number of epochs (baseline) |
| `b0_final_raw` | the same without z-scoring, as W1-12 fed it |
| `layer_0` … `layer_25` | one probe per layer (the sweep, in `layer_sweep.csv`) |
| `selected_layer` | the sweep layer with the best training-CV accuracy |
| `mixture_all` | learned softmax weights over all 26 layers, then the linear head |
| `mixture_middle` | the same over conformer layers 6–18 (predeclared in the config) |

The mixture adds 26 (or 13) weights that start uniform. They are learned with a 10× larger
learning rate (`layer_weight_learning_rate` 0.01) and no weight decay.

## Selection (training data only)

1. Five grouped, label-stratified folds over the 500 training clips. Clips sharing a recording
   ID or identical audio stay in one fold (the split-freeze grouping). Recording IDs are unknown
   for most pilot clips, so folds can still share recordings and CV may be optimistic.
2. Every arm is trained on each fold with three seeds (490, 491, 492) and scored after 30, 150,
   500 and 1,000 epochs (60 to 2,000 updates; W1-12 used 30). One run scores every budget,
   because without a scheduler the model after e epochs equals a run of exactly e epochs.
3. Each arm keeps the budget with the best mean CV accuracy (ties: macro-F1, then fewer epochs).
   `selected_layer` is the layer probe with the best CV accuracy (ties: macro-F1, then lower index).
4. `selection.json` is written. Only after that are dev labels read: each arm is refitted on
   all 500 clips with its chosen budget and three seeds, and scored on the 200 dev clips.

A test confirms that shuffling the dev labels leaves `selection.json` and every prediction
byte-identical.

## Run it

Acoustic environment (`environments/acoustic.txt`; only torch, numpy and soundfile are used),
from the repository root:

```bash
python src/experiments/layer_mixture.py \
  --output-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/runs/layer_mixture/<new-run-id>"
```

- **Colab:** mount Drive and set `OUTPUT_ROOT=/content/drive/MyDrive/Arabic-Dialect-Identification-data/output`.
  Get this code into the session as the W1-12 notebook does (pinned commit, or its ZIP mode
  while the script is not merged). A CPU runtime is enough.
- **Locally:** download the cache folder above into `<dir>/features/arabest-rq-300m-6k/dff453e2b5f7/`
  and pass `--output-root <dir>`.

Measured on an Apple M3 with one thread, on a fake cache of full pilot size: about 6 minutes
for cross-validation (435 fits) plus about 1 minute of final fits. `--threads` can speed it up,
but the default of 1 is the reproducible setting.

Exit codes: `0` completed, `1` setup or contract error (missing cache, uncached clip, bad config).

## Outputs (in the run folder)

| File | Content |
|---|---|
| `summary.json` | per headline arm: layers, epochs, updates, CV accuracy, dev accuracy and macro-F1 (mean, std, per seed), paired comparison with `b0_final` per seed (exact McNemar), shared-evaluator metrics |
| `layer_sweep.csv` | per layer: chosen epochs, CV accuracy and macro-F1, dev accuracy and macro-F1 |
| `selection.json` | every choice, written before dev labels are read |
| `mixture_weights.json` | learned layer weights per mixture arm and seed |
| `cv_results.jsonl`, `folds.json` | every CV score; fold and group of each training clip |
| `arms/<arm>/seed_<s>/` | `predictions.jsonl` (20 logits per dev clip), `head.pt`, `evaluation/` from the shared evaluator |
| `run_record.json` | commit and dirty flag, script and config hashes, manifest hashes, cache identity and cache-key hashes, timings, environment, artifact hashes |

## Reading the result

- Compare `mixture_*` and `selected_layer` with `b0_final` over all three seeds, not one.
- `b0_final` against `b0_final_raw` shows what z-scoring alone changes.
- A large mixture weight on `frontend` or the first conformer layers is a warning sign: early
  layers carry more recording-channel information, which may not hold up on Casablanca.
- Casablanca is not scored here; its features are not cached yet, and the protocol keeps it a
  frozen diagnostic, not a selection set.

## Same testing pipeline as the earlier pilots

Only training changes (layer, z-scoring, number of epochs chosen by training CV, three seeds).
Testing is the same as W1-12, MFCC, Jais and the text pilots:
- **Clips and labels:** the same 200 dev clips and 500 training clips (manifest hashes `fb8684c3…` and
  `4fb98eab…`), the same dev metadata and the same `configs/labels.json`.
- **Features:** the same cache entries as W1-12. Feature config `e4b3c287…`, provenance `dd7e012c…`,
  and train/dev cache-key hashes `e3534100…` and `03518ee5…` are all equal.
- **Evaluator:** the same `src/evaluation/evaluate.py` (revision `857d2f29…`, unchanged since it was
  added), called with the same flags. The script also stops if the evaluator's accuracy differs
  from its own.
- **Predictions:** the same row format: 20 logits in canonical order, manifest order, and ties to
  the lowest label index.
- **Denominator:** 200. W1-12 had no clip failures. This pilot refuses to run if any clip is
  uncached, rather than writing failure rows.

`SameTestingPipelineAsW112Tests` in the test file locks all of this against the committed W1-12
records.

## Checks

```bash
python -m unittest discover -s tests -p 'test_layer_mixture.py' -v
```

The tests use a tiny random encoder on synthetic audio. They cover:
- provenance identical to the encoder's own, and the pinned folder name;
- grouped, stratified, deterministic folds;
- the equivalence that lets one run score every budget;
- mixture weights, metrics, McNemar and the tie-breaks;
- an end-to-end run with the shared evaluator, a byte-identical rerun and dev-label shuffling;
- refusal of a missing cache or an uncached clip.
