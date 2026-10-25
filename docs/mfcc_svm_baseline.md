# Task 11: MFCC + SVM pilot sanity baseline

Linear: [EEC-15 / W1-11](https://linear.app/eece490/issue/EEC-15/w1-11-run-the-mfcc-plus-svm-sanity-baseline).
Purpose: a simple reference that exercises the complete data → features → model →
shared-evaluator path on the frozen pilot (`w1-07-pilot-v1-e61cd569ac7b`).

**Result role:** pilot sanity reference. It is not a competitive substitute for the
frozen-encoder or Jais baselines, and the development pilot is in-source ADI-20, not
out-of-domain.

## Run it

Colab (CPU runtime is enough): open `notebooks/W1_11_MFCC_SVM_Baseline.ipynb` and run
the cells in order. It mounts your Drive, makes sure all 700 pilot WAVs exist (fetching
any missing ones, see below), runs the script, prints the metrics and downloads a zip
of the run folder.

### Getting the pilot audio without the full dataset

Without access to the team's prepared Drive folder, fetch only the pilot clips:

```bash
python -m pip install -r requirements-data.txt
python src/data/fetch_pilot_audio.py --audio-root "$DATA_ROOT/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43"
```

Each pilot manifest row records its source shard, source row and audio SHA256. The
script opens each of the 44 shards of the pinned public release remotely, reads only
the Parquet row groups holding pilot rows (audio column only), and writes the original
WAV bytes to the same relative paths as `prepare.py` (about 0.4 GB in total). It
recomputes each clip ID from shard + row and checks byte length and SHA256 before
writing; nothing is written for a mismatching clip. It is resumable and refuses to
replace a differing existing file. Checks:
`python -m unittest discover -s tests -p 'test_fetch_pilot_audio.py' -v`.

Command line, from the repository root:

```bash
python src/experiments/mfcc_svm_baseline.py \
  --audio-root "$DATA_ROOT/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43" \
  --output-dir "$DATA_ROOT/runs/mfcc_svm_w1_11/<new-run-id>"
```

The train/dev manifests default to the committed frozen pilot
(`data/manifests/{train,dev}_pilot.jsonl` and their metadata). The configuration is
`configs/mfcc_svm_baseline.json`. Unsupported window settings are rejected; only
the documented Hamming window is implemented. Dependencies: numpy, scipy, scikit-learn, soundfile
(no GPU, no torch). The output directory must not already exist. Every WAV is checked against its
manifest `audio_sha256` (and `encoded_bytes` when supplied) inside the experiment
before decoding, including files already present on Drive. A readable file with
the wrong identity becomes an explicit `audio_error` with
`error_code=audio_integrity_mismatch`; it is never used for fitting or prediction.

Exit codes: `0` completed with no clip failures, `2` completed with explicit clip
failures (still written and scored), `1` setup or contract error.

## Features (`mfcc-stats-v1`)

Implemented in NumPy so the exact computation is visible and dependency-free:

| Step | Setting |
|---|---|
| Audio | mono (channel mean), 16 kHz (polyphase resampling if a file differs) |
| Pre-emphasis | 0.97 |
| Framing | 25 ms Hamming window, 10 ms shift, no centring padding; clips shorter than one frame are zero-padded to one frame |
| Spectrum | 512-point FFT power spectrum |
| Mel filterbank | 40 HTK-style triangular filters, 20–7600 Hz (matches `librosa.filters.mel(htk=True, norm=None)`; checked in tests) |
| Cepstrum | natural log (floor 1e-10), orthonormal DCT-II, coefficients c0–c12 |
| Dynamics | deltas and delta-deltas, regression width 2, edge replication |
| Clip summary | per-coefficient mean and standard deviation over all frames |

Result: 13 × 3 × 2 = **78 features per clip**. Feature names are saved with the model.

## Classifier and leakage controls

- `StandardScaler` + `SVC(kernel="rbf", gamma="scale", break_ties=True)`, inside one
  scikit-learn `Pipeline`.
- Week 1 tuning is deliberately small: `C ∈ {1, 10}` chosen by 5-fold stratified CV
  (seed 490) **on the training pilot only**, scoring macro-F1. The scaler is refitted
  inside every fold, then on the full training pilot for the final model.
- Development rows retain only `clip_id`, `audio_path` and audio integrity fields before feature
  extraction and prediction; gold dev labels are read only by the shared evaluator
  after `predictions.jsonl` is written. A test confirms that permuting dev labels
  leaves predictions unchanged.
- Features come from audio only: no filenames, paths, source or speaker metadata.

## Outputs (in the run folder)

| File | Content |
|---|---|
| `predictions.jsonl` | one row per dev pilot clip in manifest order, shared contract; hard labels |
| `evaluation/metrics.json`, `evaluation/confusion_matrix.csv` | shared evaluator output |
| `model.joblib` | fitted scaler + SVM, label order, feature names and feature config |
| `dev_decision_scores.jsonl` | SVM one-vs-rest decision scores in canonical label order (diagnostic; not probabilities or logits, so not written into the prediction contract) |
| `features.npz` | train/dev feature matrices and clip IDs (reruns of the classifier need no audio) |
| `failures.jsonl` | any unreadable clips with error type/message (empty when none) |
| `config.json`, `run_record.json` | config used; commit, dirty flag, input/artifact hashes, CV results, counts, timings, peak RSS, package versions |

`model_revision` is `mfcc-svm-pilot-sanity:<first 16 hex of sha256(model.joblib)>`.
Unreadable dev clips become explicit `audio_error` rows and stay in the 200-clip
denominator. Unreadable training clips are left out of fitting and counted.

## Measurement

`resources` in the run record splits wall time into training feature extraction,
fit (including CV), dev feature extraction and prediction, plus peak resident memory.
On Colab most of the time is reading WAVs from Drive; the arithmetic itself took about
6 s for all 700 clips on a CPU in a synthetic dry run.

## Checks

```bash
python -m unittest discover -s tests -p 'test_mfcc_svm_baseline.py' -v
```

Offline checks on synthetic audio include: feature shape/determinism, DCT and mel
filterbank against SciPy/librosa references, resampling and stereo handling, path
traversal, end-to-end outputs and evaluation, model replay from saved features,
dev-label isolation, explicit failure rows, refusal of reused output folders, wrong
splits and tampered manifests, and that the defaults are the frozen 500/200 pilot.
The librosa cross-check is skipped when librosa is not installed.


## Post-review safeguards

The notebook is pinned to an immutable reviewed source commit, so setup does not
require the original task branch to exist. The experiment enforces audio hashes
on both splits even when downloading is skipped. Regression tests cover a
readable, same-size replacement WAV, preserving integrity fields while stripping
gold labels, and rejecting unsupported windows.

The committed 49/200 evidence records the original pre-fix script; see its
explicit source identity in the result review. These input/config safeguards do
not change MFCC values for valid WAVs. New wall times will also include the cost
of hash verification. No replacement full-audio run is claimed by these fixes.
