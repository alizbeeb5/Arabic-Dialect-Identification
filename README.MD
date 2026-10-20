# Arabic Dialect Identification

EECE 490 research project investigating whether dialect verification
improves identification on speech from different data sources.

The initial plan uses ADI-20 for training and Casablanca for target-domain
evaluation, subject to source-overlap and checkpoint-exposure checks.

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
