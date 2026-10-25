# Training execution and storage plan

Status: execution interface documented; cluster allocation, pricing and runtime verification pending
Task: EEC-9 / W1-05

## 1. Responsibilities and delivery

### Our research team
- Develop and maintain preparation, training, inference and evaluation code.
- Provide a repository commit, dependency specifications and experiment config.
- Specify dataset/checkpoint revisions, seeds and requested experiments.
- Analyse returned results and decide subsequent experiments.

### External training team
- Provide access to its A100/H100 computing infrastructure.
- Install the supplied environment using our instructions.
- Prepare the pinned dataset once on its storage.
- Launch the requested jobs using its supported execution system.
- Return checkpoints, predictions, metrics, logs and run records.

### Delivery interface
- Code: https://github.com/alizbeeb5/Arabic-Dialect-Identification
- Each experiment identifies an exact repository commit and configuration.
- Data preparation accepts a configurable storage root.
- Instructions will provide the preparation and execution commands.
- Runtime verification status is recorded separately from the planned setup.

## 3. Configurable storage roots

The training team supplies these paths through configuration:

| Setting | Purpose |
|---|---|
| DATA_ROOT | Prepared datasets, manifests and data-audit reports |
| OUTPUT_ROOT | Experiment outputs and reusable feature/transcript caches |
| CACHE_ROOT | Optional download/library cache; defaults to DATA_ROOT/cache |

Paths are configurable. Reusable code must not contain our Drive paths
or a fixed cluster-specific directory.

### Dataset layout

| Path | Contents |
|---|---|
| DATA_ROOT/raw/ | Unchanged source Parquet shards, organized by dataset/revision |
| DATA_ROOT/audio/ | Extracted original audio |
| DATA_ROOT/manifests/ | Versioned manifests, metadata and exclusions |
| DATA_ROOT/reports/ | Data-audit reports |

Prepare each dataset release once before launching training workers.
Training reads the prepared release without modifying it.

### Experiment outputs

Each experiment writes to a unique directory:

OUTPUT_ROOT/runs/<run_id>/

This contains its configuration, checkpoints, predictions, metrics,
logs and run record.

Reusable feature/transcript caches live under OUTPUT_ROOT and identify
the dataset, checkpoint and preprocessing configuration that produced them.

## 4. Portable paths and access

- Manifest audio_path values are relative to the declared audio_root.
- Resolve a clip as Path(audio_root) / audio_path.
- Moving the dataset changes the configured/metadata root, not clip IDs
  or manifest rows.
- For shared storage, every participating node must see the same release.
- For node-local storage, prepare once and replicate the same audited
  release to each node, checking content identity.
- Preparation runs outside distributed training. Shared job-level outputs
  are written by the designated main process; worker-specific logs use
  separate names.

## 5. Persistence and capacity

- The training team confirms which storage survives job shutdown.
- Completed checkpoints, predictions, metrics and run records must be
  retained and made available to our research team.
- The source download is approximately 21.77 GB. Extracted audio, caches
  and experiment outputs require additional space.
- Record measured storage requirements after preparation.

## 6. Minimal handoff package

Each requested experiment comes with:

- An exact repository commit.
- An experiment configuration identifying the model/checkpoint revision,
  dataset and split versions, label mapping, seed and resource settings.
- A dependency specification for the requested job.
- The preparation, execution and evaluation commands.
- Expected output locations and the artifacts to return.

Use separate environments for data preparation, acoustic models and
Jais where their dependencies require it.

### Current implementation status

| Component | Status |
|---|---|
| Base environment and CPU smoke command | Implemented |
| Shared prediction validator/evaluator | Implemented |
| Portable data preparation script and dependency file | Implemented; full release prepared and audited (77,821 clips) |
| Acoustic model environment and entry point | Implemented; feature extraction and frozen-head pilot measured on Colab T4, not the external cluster |
| Jais environment and entry point | Implemented; fixed 500/200 pilot measured on Colab T4, not the external cluster |
| Training commands and experiment configurations | MFCC, frozen acoustic head and transcript TF-IDF pilots implemented; full cluster training not verified |
| Cluster launcher integration | Pending the team's execution details |

Planned interfaces are not presented as working commands.
Add exact commands and dependency pins as their implementations become
available; record where they have actually been executed.

## 7. Execution sequence for the training team

1. Check out the supplied repository commit.
2. Create the environment required for the requested job.
3. Set DATA_ROOT, OUTPUT_ROOT and optional CACHE_ROOT.
4. Run CPU preparation once, or reuse the matching audited data release.
5. Launch the supplied experiment configuration using the agreed launcher.
6. Run evaluation on its predictions using the shared evaluator.
7. Return the run directory or an accessible location containing it.

Each job specifies its requested GPU count. Record the actual hardware,
software versions, worker count and launch command in its run record.

## 8. Returned artifacts

- Experiment configuration and run record.
- Saved checkpoints.
- Prediction records, including explicit failures.
- Evaluation metrics and confusion matrix.
- Training/inference logs and measured resource usage.
- Any execution errors or deviations from the requested configuration.

## 9. Week 1 measured estimate (2026-10-25)

Full calculations, measurement scopes and evidence are in
[reports/week1_results.md](../reports/week1_results.md#compute-and-storage-estimate).
The machine-readable estimates are in
[reports/week1_analysis/summary.json](../reports/week1_analysis/summary.json).

| Operation | T4 planning estimate |
|---|---|
| Frozen features, 77,821 clips / 188.798 audio hours | 8.69 h nominal; 8.69-17.38 occupied GPU hours with an explicit 1-2x buffer |
| Cold Jais inference/transcription, all clips | 44.17-52.58 occupied GPU hours from the observed train/dev rate bracket |
| Jais with the 700 existing outputs reused | 43.77-52.11 hours for remaining clips, subject to a versioned failure/retry policy |

These are extrapolations, not full-run measurements or statistical confidence ranges. The
acoustic estimate requires a long-clip memory policy before execution. A100/H100 throughput,
multi-GPU scaling, allocation and hourly price remain unknown. Use price times occupied hours
only after measuring the intended hardware; do not combine a T4 timing with an A100 rate.
No dollar quote or paid allocation is implied.

Preparation measured 21.769 GB raw shards and 21.824 GB WAV (43.593 GB total, decimal units).
All 26 pooled float32 feature layers add a calculated 6.863 GB payload for the full dataset;
final-layer-only payload is 0.264 GB. Budget downloads, transcript cache, outputs, filesystem
overhead and headroom separately. Full transcript-cache size is not yet measured.

Reuse a pinned feature/transcription cache across head/text experiments. Parallel workers must
not repeat dataset preparation or race on shared cache files. See
[acoustic commands](acoustic_classifier.md), [Jais commands](jais_pilot.md) and
[text commands](transcript_tfidf.md). Record actual cluster timings before revising this estimate.
