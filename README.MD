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