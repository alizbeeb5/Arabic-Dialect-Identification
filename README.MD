# Arabic Dialect Identification

EECE 490 research project investigating whether dialect verification
improves identification on speech from different data sources.

The initial plan uses ADI-20 for training and Casablanca for target-domain
evaluation, subject to source-overlap and checkpoint-exposure checks.

## Week 1 results

The [pilot comparison and analysis](reports/week1_results.md) combines all four models on the
same frozen 200 development clips, with saved prediction evidence, paired errors and compute
estimates. Reproduce the analysis without a GPU, raw audio or model checkpoint:

```bash
python src/analysis/week1_results.py --output-dir /tmp/week1-analysis-review
```

These are in-source development results, not official-test or out-of-domain scores.

## Base environment

Tested locally with Python 3.14.4 on Linux.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/base-lock.txt
python -m pip check
```

## CPU smoke check

```bash
python src/smoke.py
```

Expected output includes:

```text
Synthetic accuracy: 0.75
```

Each execution saves a separate record under:

```text
runs/cpu_smoke/<run_id>/run_record.json
```

This is a synthetic setup check, not a trained speech-model result.

The record includes configuration, seed, Git commit, uncommitted-change
status, package versions, timestamps, hardware, and metrics.

## Repository structure

- configs/: experiment configurations and label mapping
- docs/: protocol, data contracts, and evaluation plan
- environments/: dependency specifications and locks
- src/data/: dataset preparation
- src/models/: model implementations
- src/evaluation/: prediction validation and metrics
- src/smoke.py: CPU setup check
- runs/: generated outputs, excluded from Git

## Model environments

Acoustic and Jais dependencies will use separate environments.
Their tested versions will be recorded when the model-loading checks
are completed.

## Data and checkpoints

Downloaded datasets, checkpoints, credentials, and generated run outputs
stay outside ordinary Git history.


## Shared evaluation

Run the evaluator checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_evaluate.py' -v
```

Evaluate the synthetic fixture:

```bash
python src/evaluation/evaluate.py \
  --manifest tests/fixtures/evaluation/references.jsonl \
  --predictions tests/fixtures/evaluation/predictions_shuffled.jsonl \
  --metadata tests/fixtures/evaluation/metadata.json \
  --model-name synthetic-fixture \
  --output-dir reports/evaluation_fixture_v2
```

Use a new output directory for each run.

Expected fixture results: accuracy 50%, supported-class macro-F1
0.583333, and coverage 75%.

Outputs:
- metrics.json: metrics, failure counts, versions, and input/code hashes.
- confusion_matrix.csv: 20 dialect columns plus failure/missing.

Predictions are matched by clip_id, independent of row order.
Explicit failures count as incorrect. Missing predictions reject a
complete evaluation; --allow-incomplete produces a clearly labeled
diagnostic that retains them in the denominator.

Macro-F1 includes labels with positive reference support, checked
against the manifest metadata. All 20 labels remain in per-class results.
C_avg remains unavailable until official scoring is implemented and verified.

The fixture contains invented examples; its audio paths do not refer
to real recordings.
## Data preparation

See [data preparation instructions](docs/data_preparation.md) for CPU setup, portable storage, resume modes and Colab usage.

## Jais-ASR inference

See [the Task 10 adapter instructions](docs/jais_adapter.md) for the pinned
model, separate environment, portable batch command, resumable cache and
Colab module check. [The smoke evidence review](reports/jais_smoke_review.md)
distinguishes notebook GPU evidence from packaged-module validation.

## Freezing the pilot splits (W1-07)

Reproduce the frozen pilot from the committed full-release manifests:

```bash
python src/data/freeze_splits.py --manifest-dir src/data/manifests
```

Expected output: `Frozen w1-07-pilot-v1-e61cd569ac7b: 500 train-pilot, 200 dev-pilot clips`.
For a new prepare.py release, point `--manifest-dir` at its manifest folder instead.

Standard library only; runs in seconds on CPU. Defaults: seed 490, 25 train clips
per label, 200 dev clips (10 per label), clips spread round-robin across recording/duplicate groups (`--max-per-group N` adds a hard cap).

Outputs (small, committed to Git):
- data/manifests/train_pilot.jsonl and dev_pilot.jsonl, each with a .metadata.json sidecar
- data/manifests/split_freeze.json: split version/hash, seeds and exact clip IDs
- reports/split_audit.md, split_cross_groups.jsonl, split_exclusions.jsonl

The upstream train/validation split is kept. Clips that share an inferred recording
ID or identical audio form one group; groups that cross train/dev are excluded from
both pilots. The dev pilot is in-source, not out-of-domain. A bounded prepare.py
pilot release is refused unless `--allow-pilot-source` is passed, and the outputs are
then marked development-only.

Both the CLI and Python entry point reject `train` and `test` as the development
split before reading input manifests. Final-test data stays outside pilot selection.

Checks:

```bash
python -m unittest discover -s tests -p 'test_freeze_splits.py' -v
```



## MFCC + SVM pilot sanity baseline (W1-11)

A CPU-only reference that exercises the full data/evaluation path: 78 MFCC summary
statistics per clip, StandardScaler + RBF SVM fitted on the training pilot only,
scored on the 200-clip dev pilot with the shared evaluator. It is a sanity
reference, not a competitive baseline. Run `notebooks/W1_11_MFCC_SVM_Baseline.ipynb`
on Colab (it fetches just the 700 pilot clips from Hugging Face if they are not on
your Drive, via `src/data/fetch_pilot_audio.py`), or:

```bash
python src/experiments/mfcc_svm_baseline.py --audio-root "$AUDIO_ROOT" --output-dir "$NEW_RUN_DIR"
```

On 2026-10-24 it scored **24.50% accuracy / 0.2383 macro-F1** on the 200-clip dev
pilot (49/200; chance 5%), in 17.7 s on CPU. See the [result review](reports/mfcc_svm_pilot_review.md)
and [the Task 11 instructions](docs/mfcc_svm_baseline.md). Checks:

```bash
python -m unittest discover -s tests -p 'test_mfcc_svm_baseline.py' -v
```

## Verified Jais pilot results

On 2026-10-24, pretrained Jais-ASR was evaluated on the frozen, balanced ADI-20
pilot. No Jais fine-tuning was performed. This is an **in-source development
result**, not official-test or out-of-domain benchmark performance.

| Development metric | Result |
|---|---:|
| Clips / dialect classes | 200 / 20 |
| Correct dialect predictions | 143 / 200 |
| Accuracy (failures count as incorrect) | **71.50%** |
| Macro-F1 (all 20 supported classes) | **70.46%** |
| Dialect-prediction coverage | **100%** |
| Transcript quality | 196 clean, 4 explicitly truncated |
| Wall time (including loading and I/O) | 486.47 s |
| Peak allocated GPU memory (Tesla T4, FP16) | 11.85 GiB |

All 200 development IDs are retained in evaluation, including the four partial
transcripts. The 500-clip training subset supplies 492 usable transcripts and
8 explicit refusals for the text-classifier handoff. C_avg is unavailable.

See the [verified result and failure analysis](reports/jais_pilot_review.md),
[exact metrics JSON](reports/jais_pilot_metrics.json), and
[Task 13 workflow](docs/jais_pilot.md). Recompute the uploaded evidence with
`python tests/check_jais_pilot_evidence.py /path/to/task13_evidence.zip`.

## Transcript-only pilot baseline (Task 14)

On the same frozen 200 development clips, combined word/character TF-IDF plus
logistic regression scored **25.00% accuracy**, **23.64% macro-F1** and **100%
prediction coverage**. It was fitted on 492 available training transcripts and
used all four partial dev transcripts. The fixed first run was not tuned after
seeing development results; it is a small-data baseline, not a benchmark claim.

See the [Task 14 result review](reports/transcript_tfidf_pilot_review.md),
[exact metrics](reports/transcript_tfidf_pilot_metrics.json), and
[reproduction instructions](docs/transcript_tfidf.md).

## Layer sweep and learned layer mixture (Idea 1 pilot)

Tests whether frozen AraBEST-RQ layers other than the final one, or a learned
mix of layers, separate dialects better than the W1-12 head. It reads only the
existing 26-layer feature cache (no encoder, GPU or audio), chooses budgets and
the best layer by grouped cross-validation on training clips, and scores dev
only afterwards.

Run1 (2026-11-19, Colab) used the frozen 500-clip training and 200-clip
development pilot with seeds 490, 491 and 492. Training cross-validation chose
conformer layer 19 and every epoch budget before dev labels were read. This is
an **in-source development result** (one clip = 0.5 points), not official-test
or out-of-domain benchmark performance.

| Arm | Layers | Epochs | Dev accuracy (mean ± std) | Dev macro-F1 |
|---|---|---:|---:|---:|
| Final layer, z-scored | final | 1000 | 47.3% ± 0.6 | 0.472 |
| Final layer, raw | final | 1000 | 47.3% ± 0.5 | 0.471 |
| Best single layer (training CV) | conformer 19 | 1000 | 79.0% ± 0.8 | 0.790 |
| All-layer mixture | all 26 | 500 | 82.8% ± 1.0 | 0.827 |
| Middle-layer mixture | conformer 6–18 | 500 | **83.2% ± 0.6** | **0.830** |

The team's 2026-11-20 handoff review rescored all 15 heads (5 arms × 3 seeds)
with the shared evaluator. The middle-layer mixture is the main acoustic
candidate for full-data training. Checkpoints, dev predictions and the run
record are on the team Google Drive under
`Arabic-Dialect-Identification-data/output/runs/layer_mixture/run1/`, not in
this repository. See the [method and run instructions](docs/layer_mixture.md).

## Casablanca check of the Idea 1 heads

Scores the frozen Idea 1 heads, unchanged, on the frozen Casablanca validation
split (6,819 TV-drama clips, 8 dialects) as a cross-dataset diagnostic. Casablanca
labels choose nothing. Every head must first reproduce its saved ADI-20 dev
logits. It has not been run yet. See the
[method and Colab steps](docs/casablanca_check.md).
