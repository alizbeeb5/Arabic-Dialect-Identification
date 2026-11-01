# Train the three Arabic dialect classifiers

**Team:** ALI ZBEEB, FATIMA KARNIB, HAIDAR MADI  
**Task:** EEC-23 / W2-01  
**Repository:** https://github.com/alizbeeb5/Arabic-Dialect-Identification  
**Delivery branch / PR:** `feat/eec-23-acoustic-a` / #18. Use the exact commit supplied with the handoff, not an unpinned future branch head.

## 1. What to train

| Model | Architecture | Trainable parts | Configuration |
|---|---|---|---|
| A | AraBEST-RQ → masked mean → linear 848 → 20 | Linear head | `configs/acoustic/A.json` |
| B | AraBEST-RQ → masked attention → linear 848 → 256 → GELU → dropout 0.2 → linear 256 → 20 | Attention + MLP; encoder frozen/eval | `configs/acoustic/B.json` |
| C | Exactly B's pooling and MLP | CNN/Conformer encoder + attention + MLP | `configs/acoustic/C.json` |

B/C share `src/models/acoustic_architectures.py` (331,284 attention/MLP parameters in either arm; C additionally trains encoder parameters). Attention scores use `Linear(848,128) → tanh → Linear(128,1)` with a masked temporal softmax. Their common trainer/inference interface is `src/experiments/acoustic_abc.py`; A delegates to the established cached-feature runner. `scripts/train_acoustic.sh` launches any arm or all three. The existing A-only files remain for compatibility; the new shared files are the starting point.

All arms use the same pretrained `Elyadata/AraBEST-RQ-300M-6k` revision `dff453e2b5f72f76211a523b1050691967e5a59b`, 16 kHz mono full clips, multiclass cross-entropy and all 20 labels in `configs/labels.json`. B/C attend only to real frames. B uses ordinary no-grad encoder frames, so attention can learn. C uses a differentiable encoder forward and never detached caches. No transcript model or verifier is trained here.

## 2. Install and locate data

Linux, Python 3.12, one GPU per process. No multi-GPU launcher is required. Independent arms may be scheduled on separate GPUs using `CUDA_VISIBLE_DEVICES`; keep their output directories separate. Cluster throughput/memory has not been measured.

```bash
git clone --branch feat/eec-23-acoustic-a https://github.com/alizbeeb5/Arabic-Dialect-Identification.git
cd Arabic-Dialect-Identification
# Check out the exact commit from the handoff message before running.
git rev-parse HEAD
python3.12 -m venv .venv-acoustic
source .venv-acoustic/bin/activate
python -m pip install -r environments/acoustic.txt
python -m pip check

export DATA_ROOT=/absolute/path/Arabic-Dialect-Identification-data
export RUNS_ROOT=/absolute/path/retained/acoustic-runs
export DEVICE=cuda
```

Use the prepared ADI-20 release `full-8db5842080c2f643`: **67,015 train / 10,806 source dev**. Full manifests and sidecars are committed in `src/data/manifests/`; audio is outside Git. Default WAV location:

`$DATA_ROOT/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43/<audio_path>`

Set `AUDIO_ROOT` if your identical audio tree is elsewhere. `CACHE_ROOT` controls checkpoint downloads; `OUTPUT_ROOT` controls A's reusable pooled feature cache. Put caches on local fast disk where practical and keep run/checkpoint storage durable. B computes frames on the fly rather than requiring a frame cache; C cannot use detached frame caches. Preflight reports an approximate frame-cache storage size for comparison with measured on-the-fly timing.

If the prepared data is unavailable, follow `docs/data_preparation.md` to obtain the pinned release. Do not substitute another dataset, pilot, or label order. Do not edit manifests to hide missing files. Installation errors and resource failures should be returned with their logs.

## 3. Run these stages

First run the engineering smoke checks:

```bash
bash scripts/train_acoustic.sh all smoke
```

A retains its existing 500/200 pilot smoke. B/C use the same deterministic short, complete-clip engineering subset: 40 train / 20 dev IDs selected from the approved release. The subset manifests are saved in each run. Smoke results are not the full experiment.

Then run the full-data memory checks, followed by one selection run per arm:

```bash
bash scripts/train_acoustic.sh all preflight
bash scripts/train_acoustic.sh all select
```

Preflight actually runs the longest full train/dev inputs (source dev reaches 212.38 seconds). For B/C it includes a training backward pass **and AdamW update**, so optimizer state allocation is measured. Weights are restored before real training. A preflight checks its frozen extraction path. Every full job repeats preflight before training. If a check fails, return the failure record and stop; do not add unrecorded cropping/windowing or change precision. A common revised policy must be specified across all arms before another comparison.

Each selection run uses **seed 11**, the fixed config and the last epoch. Please return those results first. After we confirm the configuration, run:

```bash
bash scripts/train_acoustic.sh all report
```

This runs **17, 29, 43** for each architecture. The initial request is 3 selection + 9 reporting full runs, **12 total**. At most one additional selection configuration per architecture is allowed under the existing 15-run ceiling; none is automatically launched. Configurations are compared using source-dev macro-F1, then accuracy, then lower learning rate. Keep all reporting seeds; do not pick the best seed. Casablanca validation is scored later as a separate diagnostic, never used for this selection.

For one arm only, replace `all` with `A`, `B` or `C`.

### Fixed budgets

| Setting | A | B | C |
|---|---|---|---|
| Full epochs / original-clip exposure | 30 / 2,010,450 if all train clips work | Same | Same |
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | 0.01 weights; 0 bias | 0.01 matrix weights; 0 bias/norm | Same as B |
| Head learning rate | 0.001 | 0.001 | 0.001 |
| Encoder learning rate | Frozen | Frozen | 0.00001 |
| Effective batch | 256 cached vectors | 32 clips | 32 clips |
| Encoder microbatch | 1 at extraction | 1, gradient accumulation | 1, gradient accumulation |
| Precision / clip policy | FP32 / full | Same | Same |
| Checkpoint selection | Last epoch | Same | Same |

The number of updates differs because A's head is inexpensive: 7,860 A updates versus 62,850 B/C updates when all clips work. Original-clip exposure and selection opportunities are matched. Do not interpret B−A as attention alone: pooling and classifier capacity both change. C−B tests encoder adaptation with identical pooling/head capacity. No scheduler, augmentation, class rebalancing or automatic hyperparameter search is used. Training-only overfit checks run before fitting; their temporary weights are discarded. Inspect training loss/gradients before treating a 30-epoch result as adequately optimized.

### Interrupted B/C job

Each completed epoch atomically replaces `checkpoint.pt`, including optimizer state, RNG state and curve history. Resume from the last completed epoch; an interrupted partial epoch is repeated. Use a new run directory:

```bash
python -m src.experiments.acoustic_abc train --architecture C --stage select \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
  --resume /absolute/path/previous-run/checkpoint.pt
```

Resume requires identical architecture/config, source files, manifests and retained training IDs. A resumes frozen feature extraction through its cache; its small CPU head restarts from the fixed seed. Failed jobs keep `failure.json`/`run_record.json` for diagnosis.

## 4. What to return

Send the complete `$RUNS_ROOT` directory (or an archive), including failed attempts and `logs/`. For each architecture/seed return:

- `checkpoint.pt`: trained model and B/C epoch-resume state. B checkpoints contain attention/MLP weights; their frozen encoder is identified by pinned revision/hashes. C checkpoints also include trained encoder weights.
- `config.json` and `learning_curve.jsonl`: effective settings and train/dev progress.
- `predictions.jsonl`: one row for every evaluated dev ID, all 20 canonical logits for successful rows, explicit failures otherwise.
- `evaluation/metrics.json` and `evaluation/confusion_matrix.csv`: accuracy, macro-F1, per-dialect results, coverage and failures.
- `run_record.json`, `overfit_check.json`, `preflight.json` for full runs, `failures.jsonl`, and console logs. Include the preparation/checkpoint/source/config identities.
- `python -m pip freeze`, `python -m pip check` and `nvidia-smi` output. Report GPU model/count actually used, runtime, peak allocated memory, disk usage and actual cost if known.

Audio failures remain in the full dev denominator; do not silently drop or replace examples. A partial/failed job is returned as such. Do not upload raw audio, downloaded pretrained weights or generated training runs into Git.

## 5. Reload without training

The same command loads A, B or C from the returned checkpoint:

```bash
python -m src.experiments.acoustic_abc predict \
  --checkpoint /absolute/path/run/checkpoint.pt \
  --manifest src/data/manifests/dev.jsonl \
  --data-root "$DATA_ROOT" --output-dir /absolute/new/prediction-check
```

For Casablanca, supply its frozen manifest and the appropriate `--audio-root`; retain all 20 candidates and all reference IDs. Human transcripts/source metadata do not enter the model. Use the exact training source and pinned environment. Independent reproduction requires identical labels and logits within absolute tolerance **1e-6**, rtol 0 on the same hardware/settings. Measure and record differences before changing tolerances on other hardware. Prediction does no fitting or checkpoint selection.

## Validation scope

Offline tests exercise tiny random Conformer models with synthetic audio, real gradient updates, frozen-state checks, mask invariance, end-to-end training/evaluation, checkpoint-only prediction, explicit failure denominators and exact epoch-boundary optimizer/RNG resume. These do **not** establish pretrained 300M CUDA correctness, actual long-clip memory fit, convergence or dialect accuracy. Real-checkpoint smoke and long-input checks are the team's first execution steps. Keep EEC-23 In Progress pending real runs and independent review.
