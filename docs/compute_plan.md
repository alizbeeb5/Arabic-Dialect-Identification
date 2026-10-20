# Training execution and storage plan

Status: draft
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
| Portable data preparation script and dependency file | Pending Task 6 |
| Acoustic model environment and entry point | Pending model implementation |
| Jais environment and entry point | Pending adapter implementation |
| Training commands and experiment configurations | Pending training implementation |
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