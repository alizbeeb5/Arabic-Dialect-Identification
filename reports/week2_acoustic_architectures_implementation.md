# EEC-23: A/B/C implementation and delivery validation

Authors: ALI ZBEEB, FATIMA KARNIB, HAIDAR MADI  
Date: 2026-11-01  
Status: implementation packaged; real pretrained GPU smoke, full training and independent review pending.

The delivery entry point is [TRAINING_TEAM.md](../TRAINING_TEAM.md). This supersedes the A-only handoff for new training-team work, while preserving the old A runner, config paths, sample and saved-head inference interface.

## Architecture and organization

- `src/models/acoustic_architectures.py`: one masked additive-attention module and one shared 848 → 256 → GELU → dropout(0.2) → 20 MLP. Attention uses a 128-unit tanh scorer. B/C each have 331,284 attention/head parameters; C additionally unfreezes the loaded CNN/Conformer model.
- `src/models/acoustic_encoder.py`: shared frame forward now exposes a differentiable final-layer path. Existing inference-mode pooled-cache APIs retain their output contract. B runs final frames under `no_grad` so attention backward remains valid; C retains encoder autograd. Sentence normalization and fbank settings stay fixed.
- `configs/acoustic/{A,B,C}.json`: canonical full-run configurations. A's legacy config remains compatible; B/C use identical heads, optimizer settings, effective batches and clip exposure, differing in encoder freezing.
- `src/experiments/acoustic_abc.py`: shared train/predict CLI; A delegates to its existing runner. B/C use complete clips and one-clip microbatches with accumulation to 32. Audio hashes, manifests/sidecars and label order are checked. All dev IDs and explicit failures are preserved.
- `scripts/train_acoustic.sh`: portable A/B/C/all launcher with logs and smoke/preflight/select/report stages. No cluster scheduler, multi-GPU speedup or target-machine memory capacity is assumed.

## Training and recovery

The approved full release is 67,015 train / 10,806 dev. Full runs are 30 epochs: 2,010,450 original-clip exposures per architecture if all training audio works. A has 7,860 updates at batch 256; B/C have 62,850 updates at effective batch 32. Weight decay is 0.01 for matrix weights and zero for bias/norm parameters. B/C head LR is 0.001; C encoder LR is 0.00001. All use FP32, the last epoch, no scheduler, augmentation or class rebalancing. Equal selection opportunities and the original 15-run ceiling are retained: one seed-11 config plus reporting seeds 17/29/43 initially requests 12 full jobs total. Additional configurations are not launched automatically.

B/C longest-input preflight executes a training forward/backward/AdamW step and dev inference, recording failures and measured time/memory. Temporary weights and optimizer moments are discarded before training. The training-only small-sample overfit gate also restores initial weights afterward. Preflight reports a storage estimate for full FP32 final frames; the selected implementation computes frames on the fly rather than relying on a pooled/frame cache. The estimate is not a measured cluster storage or throughput result.

B freezes the encoder and verifies unchanged state, zero encoder gradients and trained pooling/head gradients. C records nonzero encoder gradients and separate parameter-only hashes to show actual parameter updates rather than buffer changes alone. Full run artifacts include configuration, source/input/encoder provenance, curves, canonical logits, evaluator metrics/confusion, failures, runtime/memory/storage and checkpoints.

Each B/C epoch checkpoint includes model, optimizer, curve and RNG state. Resume requires matching source hashes, config, manifest/sidecar/label hashes and retained training IDs, and resumes after the last completed epoch. Prediction reloads any returned A/B/C checkpoint without fitting; the common interface permits labeled dev diagnostics and rejects test prediction.

## Offline validation

Pinned acoustic environment installed successfully; `python -m pip check` reports no broken requirements. Torch 2.14.1+cu130, torchaudio 2.11.0+cu130 and SpeechBrain 1.1.2 imported successfully. Tests ran on CPU with tiny random Conformer models and synthetic audio, not the released 300M weights.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
bash -n scripts/train_acoustic.sh scripts/run_acoustic_A.sh
python -m py_compile src/experiments/acoustic_abc.py src/models/acoustic_architectures.py src/models/acoustic_encoder.py
python -m pip check
git diff --check
```

The full unittest suite ran **183 tests, with one optional reference check skipped and no failures**. Twelve dedicated architecture checks cover:

- attention masking including NaN padding and empty-mask rejection;
- identical B/C pooling/head initialization, frozen B state, real C gradients/parameter updates;
- final-frame equivalence with the existing frozen inference path;
- fixed B/C architecture/config capacity and matched epoch exposure;
- end-to-end B/C training, all-20 score exports, shared evaluation and checkpoint-only inference within absolute tolerance 1e-6;
- exact C parameter equality after a simulated stopped job and optimizer/RNG resume;
- longest-input preflight performing a real parameter update and durably recording simulated OOM;
- explicit audio failure coverage/denominators;
- rejecting pilot data as a full result before model loading;
- stopping and preserving evidence when the overfit gate fails;
- unchanged C weights when development labels are permuted;
- rejecting test prediction before A/B/C loading.

## Limits and next execution step

No actual pretrained-checkpoint CUDA smoke, full-length 300M memory fit, full-data optimization, dialect accuracy, cluster throughput or cost is claimed. Those checks require the training machine and dataset. The team first runs the real smoke and full longest-input preflight, then seed-11 selection jobs. Preserve failure evidence and record any revised common memory/clip/optimization policy before new results. Return all checkpoints and reporting seeds for source-dev comparison and later Casablanca diagnostics. EEC-23 remains In Progress until returned runs, comparisons and independent review are complete.
