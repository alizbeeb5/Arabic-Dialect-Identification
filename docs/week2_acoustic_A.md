# Architecture A — training-team handoff

**Project:** EECE 490 — Arabic Dialect Identification / NADI ADI-20
**Team:** ALI ZBEEB, FATIMA KARNIB, HAIDAR MADI
**Linear:** EEC-23 / W2-01
**Scope:** implement, check and train architecture A only.

## Model and research question

16 kHz mono audio → frozen AraBEST-RQ 300M → final encoder frames → masked temporal mean → `Linear(848, 20)` → 20 logits.

The encoder is `Elyadata/AraBEST-RQ-300M-6k`, revision
`dff453e2b5f72f76211a523b1050691967e5a59b`. The loader verifies SHA256 of
`model.ckpt` and `normalizer.ckpt` and strictly loads weights. It uses sentence
normalization, full attention and FP32 extraction. Checkpoint hashes are recorded
in each run. The encoder stays in evaluation mode with zero trainable parameters;
its state hash must be unchanged and the head reads detached cached features.

Only the **16,980 head parameters** are trained. There is no hidden layer,
attention pooling, transcript branch or verifier. A is the reference for the later
B/C comparisons. No benchmark-win claim follows from in-source development scores.

Final layer means representation index **25** (24 Conformer layers plus frontend
and final normalization output). Only that layer is saved; the previous all-layer
pilot cache is a different feature configuration and is not silently reused.
The current encoder still computes its normal forward path before selecting it.

## Files

| File | Purpose |
|---|---|
| `src/models/acoustic_encoder.py` | Existing verified frozen encoder, masked pooling and content-keyed cache |
| `src/experiments/acoustic_classifier.py` | Training, integrity checks, preflight, predictions and evaluation |
| `configs/week2_acoustic_A_smoke.json` | 500/200 engineering sample |
| `configs/week2_acoustic_A.json` | Approved full-release contract and training settings |
| `scripts/run_acoustic_A.sh` | Portable launch commands |
| `src/experiments/acoustic_A_predict.py` | Independent inference from a saved head |
| `notebooks/W2_01_Acoustic_A.ipynb` | Colab sample; added alongside this handoff |

## Environment and hardware

Use Linux and Python 3.12 in an isolated environment, separate from Jais.

```bash
python3.12 -m venv .venv-acoustic
source .venv-acoustic/bin/activate
python -m pip install -r environments/acoustic.txt
python -m pip check
python -c 'import torch, torchaudio, speechbrain; print(torch.__version__, torchaudio.__version__, speechbrain.__version__); print(torch.cuda.is_available())'
```

The dependency specification pins torch 2.14.1, torchaudio 2.11.0 and SpeechBrain
1.1.2. Confirm CUDA/driver compatibility on the target machine; save the actual
resolved environment and GPU/driver information. Do not silently change packages
or model precision after collecting selection results. If the package setup
fails, return the installation log and the proposed environment change.

Request **one GPU** for extraction and CPU/RAM for the linear head. This runner
is single-process; reserving ten GPUs does not accelerate it. Head fitting runs
on one CPU thread, in FP32, and is cheap compared with encoder extraction.
No target-cluster timing or dollar estimate has been measured.

## Data and portable paths

Use the already-prepared dataset, **not a fresh pilot download**:

- Dataset revision: `c8da454c9f474bb075d3fe608940ee29e8ccfa43`
- Audited release: `full-8db5842080c2f643`
- Train: **67,015** clips; source dev: **10,806** clips.
- Train SHA256: `56a75aa929e2a2c4592012954386cb3d91ff7f256a7004ebe7c4a776573ea4ba`
- Dev SHA256: `525ecea4b39c504b28858ddbfaf0a954af703c91ae8edddb252a0a6f2c8d138b`

Committed full manifests and metadata are under `src/data/manifests/`; audio is
not included. The script verifies manifest hashes, sidecars, labels, per-audio
hashes and encoder provenance. It refuses a 500/200 pilot under the full config.
Full upstream manifests retain source-independence limitations documented in
their metadata; do not call these speaker-disjoint or out-of-domain results.

```bash
export DATA_ROOT=/absolute/path/Arabic-Dialect-Identification-data
export AUDIO_ROOT="$DATA_ROOT/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43"
export OUTPUT_ROOT=/absolute/local/path/acoustic-A-features
export CACHE_ROOT=/absolute/local/path/huggingface
export RUNS_ROOT=/absolute/path/retained/week2-acoustic-A-runs
export DEVICE=cuda
```

`audio_path` in the manifest is relative to AUDIO_ROOT. The old absolute Drive
path in a sidecar is provenance; the explicit AUDIO_ROOT controls loading.
Keep model/feature caches on local disk when possible. Retain the run outputs.
Set MANIFEST_ROOT only if using an identical copy of the full manifest directory.
Use the existing `prepare.py` workflow if audio must be prepared; do not alter
clip IDs, bytes or manifests to make missing files disappear.

Canonical label order:
`MSA, BAH, TUN, ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SUD, SYR, UAE, YEM`.

## Execution order

From the pinned repository checkout (or `source/` in the handoff ZIP):

```bash
python -m unittest discover -s tests -p 'test_acoustic_encoder.py'
python -m unittest discover -s tests -p 'test_acoustic_classifier.py'
python -m unittest discover -s tests -p 'test_week2_acoustic_A.py'
python -m unittest discover -s tests -p 'test_evaluate.py'
bash scripts/run_acoustic_A.sh overfit
bash scripts/run_acoustic_A.sh smoke
bash scripts/run_acoustic_A.sh preflight
bash scripts/run_acoustic_A.sh select
```

The overfit gate uses two **training** examples per class, 300 epochs, LR 0.001.
It requires 100% training accuracy and final loss at most 5% of initial loss.
If it fails, inspect the features, label joins and optimizer before full jobs.
The engineering sample fits 500 clips and evaluates 200; its 60 updates are
**not** the final baseline or evidence of optimization adequacy.

Full runs automatically repeat the long-clip preflight on the longest train
and dev recordings, using real audio, no cache shortcut, one clip per forward.
Dev reaches about **212.38 seconds**. `preflight.json` records time, memory and
failures. Any preflight failure stops before full extraction. No truncation or
chunking fallback is applied. If full-context extraction cannot fit, send the
failure record so a common A/B/C long-clip policy can be recorded before reruns.

Each command creates a new timestamped output directory and console log.
Exit **0** means success; **2** means completed with explicit clip failures;
**1** means setup/gate failure. The shell stops on any nonzero result. Preserve
failed outputs; resume extraction with the same cache and a **new run directory**.
Cache extraction resumes; an interrupted head fit restarts from its fixed seed.

## Declared training budget and selection

| Setting | Initial configuration |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| Learning rate | 0.001 |
| Weight decay | weights 0.01; bias 0 |
| Gradient clipping | norm 1.0 |
| Head batch size | 256; keep final partial batch |
| Epochs | 30 |
| Full-data updates | **7,860** when all 67,015 clips succeed |
| Original training-clip exposure | **2,010,450** when all training clips succeed |
| Scheduler / augmentation / rebalancing | none |
| Checkpoint | last epoch, fixed before running; no early stopping |
| Selection data | ADI-20 full dev; Casablanca is separate diagnostic only |

One configuration is declared initially; do not automatically search a second.
Inspect **training-only** curves for failed or inadequate optimization. Update
count alone does not establish convergence. Record any revised budget/config
before another run. At most two selection configurations are allowed for A,
using seed **11**; select by dev macro-F1, then accuracy, then lower LR on ties.
This development selection must be disclosed in the report.

After the configuration is frozen and the selection evidence is reviewed:

```bash
bash scripts/run_acoustic_A.sh report
```

This runs fresh seeds **17, 29, 43**, sequentially, reusing the verified frozen
feature cache. Report all seeds and mean/std; do not headline the best seed.
The initial plan is four full A runs (one selection + three reporting). The
shared A/B/C budget is at most 15 full runs. B/C must match original-clip exposure
and get equal selection opportunities; architecture-driven accumulation changes
must be disclosed. A failures or incomplete data are not a complete comparison.

## Independent prediction reproduction

A different teammate must rerun inference from a returned checkpoint, recording
identity, command and comparison tolerance. This command performs no fitting:

```bash
python src/experiments/acoustic_A_predict.py \
  --checkpoint /absolute/run/checkpoint.pt \
  --manifest src/data/manifests/dev.jsonl \
  --metadata src/data/manifests/dev.metadata.json \
  --audio-root "$AUDIO_ROOT" --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" --device cuda \
  --output-dir /absolute/new/prediction-review
```

Using the same verified cached features, require identical predicted labels and
logits within absolute tolerance **1e-6**, rtol 0. If re-extracting on different
hardware, first measure feature/logit differences and record the hardware and
observed tolerance; never silently loosen a failing check. The same prediction
entry point can later consume a frozen labeled Casablanca manifest, without
fitting transforms or using its results to tune A this week.

## Return package

Return **all** selection/reporting run directories, preflight/gate failures and
sibling `*.console.log` files, plus:

- `checkpoint.pt`, effective `config.json`, `learning_curve.jsonl`;
- `predictions.jsonl` with all 20 canonical logits, `failures.jsonl`;
- `evaluation/metrics.json` and `evaluation/confusion_matrix.csv`;
- `overfit_check.json`, `preflight.json`, `run_record.json`;
- exact source commit, `python -m pip freeze`, `python -m pip check`,
  `nvidia-smi` output, GPU count used/type, elapsed time, peak allocated memory,
  cache/disk size and any actual cost (unknown cost stays unknown);
- different-teammate reproduction record (name, command, tolerance and result).

Keep the feature cache available for reruns; transferring it is optional.
Full pooled final-layer vectors alone are about 252 MiB before container/index
overhead; measured storage is authoritative. Raw audio and checkpoint downloads
need additional space. Do not attach downloaded model weights or raw audio to
ordinary Git commits.

The shared evaluator retains the full dev denominator and counts failed clips
as incorrect. Return accuracy, macro-F1, coverage and per-dialect errors. C_avg
remains unimplemented/unverified. These are in-source dev measurements, not
an official-test submission.
