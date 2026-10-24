#!/usr/bin/env python3
"""W1-11 / EEC-15: MFCC summary statistics + SVM pilot sanity baseline (CPU only).

Pipeline
  1. Read every pilot WAV, compute 13 MFCCs (+ deltas, + delta-deltas) per frame and
     summarise each clip as the per-coefficient mean and standard deviation (78 dims).
  2. Fit StandardScaler + RBF SVC on the frozen *training* pilot only. The only tuning
     is a 2-value C grid chosen by 5-fold stratified CV inside the training pilot.
  3. Predict the frozen development pilot. Development labels are removed before any
     dev feature or prediction code runs, so they cannot influence fitting.
  4. Write predictions in the shared contract and score them with the shared evaluator.

This is a pilot sanity reference that exercises the data/evaluation path. It is not a
competitive substitute for the learned acoustic or Jais baselines.

Exit codes: 0 completed with no clip failures, 2 completed with explicit clip failures,
1 setup/contract error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate as ev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/mfcc_svm_baseline.json"
MODEL_NAME = "mfcc-svm-pilot-sanity"
RESULT_ROLE = "pilot_sanity_reference_not_competitive_baseline"
FEATURE_STATS = ("mean", "std")


# ---------------------------------------------------------------- small helpers

def utc_now():
    return datetime.now(timezone.utc)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def peak_rss_mib():
    # ru_maxrss is KiB on Linux and bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def progress(message):
    print(message, flush=True)


# ---------------------------------------------------------------- configuration

def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    feats, clf = config["features"], config["classifier"]
    if feats["statistics"] != list(FEATURE_STATS):
        raise ValueError(f"Only statistics {list(FEATURE_STATS)} are implemented")
    if not 0 < feats["f_min_hz"] < feats["f_max_hz"] <= feats["sample_rate_hz"] / 2:
        raise ValueError("Mel range must satisfy 0 < f_min < f_max <= Nyquist")
    if feats["n_fft"] < round(feats["frame_length_ms"] * feats["sample_rate_hz"] / 1000):
        raise ValueError("n_fft must be at least the frame length in samples")
    if not 1 <= feats["n_mfcc"] <= feats["n_mels"]:
        raise ValueError("n_mfcc must be between 1 and n_mels")
    if not clf["C_grid"] or any(c <= 0 for c in clf["C_grid"]):
        raise ValueError("C_grid must contain positive values")
    if clf["kernel"] != "rbf" or clf["model"] != "sklearn.svm.SVC":
        raise ValueError("This baseline only implements an RBF sklearn.svm.SVC")
    return config


def feature_dim(feature_config):
    blocks = 1 + bool(feature_config["include_deltas"]) + bool(feature_config["include_delta_deltas"])
    return feature_config["n_mfcc"] * blocks * len(FEATURE_STATS)


def feature_names(feature_config):
    blocks = ["mfcc"]
    if feature_config["include_deltas"]:
        blocks.append("delta")
    if feature_config["include_delta_deltas"]:
        blocks.append("delta2")
    start = 0 if feature_config["include_c0"] else 1
    coeffs = [f"c{i}" for i in range(start, start + feature_config["n_mfcc"])]
    return [f"{stat}_{block}_{c}" for stat in FEATURE_STATS for block in blocks for c in coeffs]


# ---------------------------------------------------------------- MFCC features

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    """HTK-style triangular filters on the power spectrum, shape [n_mels, n_fft//2 + 1]."""
    bins = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins[None, :] - lower) / (centre - lower)
    falling = (upper - bins[None, :]) / (upper - centre)
    return np.maximum(0.0, np.minimum(rising, falling))


def dct_ii_ortho(n_out, n_in):
    """Orthonormal DCT-II basis, shape [n_out, n_in]; matches scipy.fft.dct(norm='ortho')."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    basis[0] /= np.sqrt(2.0)
    return basis


def deltas(features, width):
    """Standard regression deltas over time with edge replication; features [C, T]."""
    padded = np.pad(features, ((0, 0), (width, width)), mode="edge")
    frames = features.shape[1]
    numerator = sum(n * (padded[:, width + n:width + n + frames] - padded[:, width - n:width - n + frames])
                    for n in range(1, width + 1))
    return numerator / (2.0 * sum(n * n for n in range(1, width + 1)))


class MfccExtractor:
    def __init__(self, feature_config):
        self.config = dict(feature_config)
        rate = self.config["sample_rate_hz"]
        self.frame_length = int(round(self.config["frame_length_ms"] * rate / 1000))
        self.frame_shift = int(round(self.config["frame_shift_ms"] * rate / 1000))
        self.window = np.hamming(self.frame_length)
        self.filters = mel_filterbank(rate, self.config["n_fft"], self.config["n_mels"],
                                      self.config["f_min_hz"], self.config["f_max_hz"])
        start = 0 if self.config["include_c0"] else 1
        self.dct = dct_ii_ortho(start + self.config["n_mfcc"], self.config["n_mels"])[start:]
        self.dim = feature_dim(self.config)

    def frames(self, wave):
        if len(wave) < self.frame_length:
            wave = np.pad(wave, (0, self.frame_length - len(wave)))
        count = 1 + (len(wave) - self.frame_length) // self.frame_shift
        index = np.arange(self.frame_length)[None, :] + self.frame_shift * np.arange(count)[:, None]
        return wave[index]

    def mfcc(self, wave):
        """Per-frame coefficients, shape [n_mfcc, frames]."""
        wave = np.asarray(wave, dtype=np.float64)
        emphasised = np.append(wave[0], wave[1:] - self.config["pre_emphasis"] * wave[:-1])
        spectrum = np.fft.rfft(self.frames(emphasised) * self.window, n=self.config["n_fft"], axis=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2) / self.config["n_fft"]
        log_mel = np.log(np.maximum(power @ self.filters.T, self.config["log_floor"]))
        return self.dct @ log_mel.T

    def summary(self, wave):
        coefficients = self.mfcc(wave)
        blocks = [coefficients]
        if self.config["include_deltas"]:
            blocks.append(deltas(coefficients, self.config["delta_width"]))
        if self.config["include_delta_deltas"]:
            blocks.append(deltas(deltas(coefficients, self.config["delta_width"]),
                                 self.config["delta_width"]))
        stacked = np.concatenate(blocks, axis=0)
        vector = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
        if vector.shape != (self.dim,) or not np.isfinite(vector).all():
            raise ValueError("Feature vector has the wrong shape or nonfinite values")
        return vector.astype(np.float64)


def resolve_audio_path(audio_root, audio_path):
    relative = PurePosixPath(audio_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"audio_path must stay under audio_root: {audio_path}")
    return Path(audio_root) / relative


def load_wave(path, sample_rate):
    """Mono float waveform at sample_rate; raises on unreadable, empty or nonfinite audio."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if rate != sample_rate:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(rate), int(sample_rate))
        wave = resample_poly(wave, sample_rate // divisor, int(rate) // divisor)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Empty or nonfinite waveform")
    return wave, int(rate)


def extract_split(rows, audio_root, extractor, split_name):
    """Return (matrix, clip_ids, failures). Uses audio only; never reads labels."""
    vectors, ids, failures = [], [], []
    started = time.perf_counter()
    for number, row in enumerate(rows, start=1):
        try:
            wave, _ = load_wave(resolve_audio_path(audio_root, row["audio_path"]),
                                extractor.config["sample_rate_hz"])
            vectors.append(extractor.summary(wave))
            ids.append(row["clip_id"])
        except Exception as error:  # noqa: BLE001 - every clip failure is recorded explicitly
            failures.append({"clip_id": row["clip_id"], "error_code": "audio_unreadable",
                             "error_type": type(error).__name__, "message": str(error)[:300]})
        if number % 50 == 0 or number == len(rows):
            progress(f"[{split_name}] features {number}/{len(rows)} "
                     f"({len(failures)} failed, {time.perf_counter() - started:.1f}s)")
    matrix = np.vstack(vectors) if vectors else np.empty((0, extractor.dim))
    return matrix, ids, failures


# ---------------------------------------------------------------- inputs

def load_split(manifest, metadata, labels_path, labels):
    rows = ev.read_jsonl(manifest)
    splits = {row.get("split") for row in rows}
    if len(splits) != 1:
        raise ValueError(f"{manifest}: expected exactly one split, found {sorted(map(str, splits))}")
    split = splits.pop()
    references = ev.validate_references(rows, labels, split)
    if len(references) != len(rows):
        raise ValueError(f"{manifest}: duplicate or unlabeled rows")
    supported = [c for c in labels if any(r["label"] == c for r in rows)]
    meta = ev.load_evaluation_metadata(metadata, manifest, labels_path, split, supported)
    return rows, split, meta


def without_labels(rows):
    """Development rows as seen by feature extraction and prediction: no gold label."""
    return [{"clip_id": r["clip_id"], "audio_path": r["audio_path"]} for r in rows]


# ---------------------------------------------------------------- model

def fit_classifier(train_x, train_y, classifier_config, seed):
    """Fit scaler + SVC on training data only. Returns (pipeline, cv_summary)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=classifier_config["kernel"], gamma=classifier_config["gamma"],
                    decision_function_shape=classifier_config["decision_function_shape"],
                    break_ties=classifier_config["break_ties"], random_state=seed)),
    ])
    grid = [float(c) for c in classifier_config["C_grid"]]
    if len(grid) == 1:
        pipeline.set_params(svm__C=grid[0]).fit(train_x, train_y)
        return pipeline, {"selected_C": grid[0], "cv": None}

    counts = np.unique(train_y, return_counts=True)[1]
    folds = min(classifier_config["cv_folds"], int(counts.min()))
    if folds < 2:
        raise ValueError("Every training label needs at least 2 clips for cross-validation")
    search = GridSearchCV(
        pipeline, {"svm__C": grid}, scoring=classifier_config["cv_selection_metric"],
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), refit=True, n_jobs=1)
    search.fit(train_x, train_y)
    results = [{"C": float(c), "mean_cv_" + classifier_config["cv_selection_metric"]: float(m),
                "std": float(s)}
               for c, m, s in zip(search.cv_results_["param_svm__C"],
                                  search.cv_results_["mean_test_score"],
                                  search.cv_results_["std_test_score"])]
    return search.best_estimator_, {"selected_C": float(search.best_params_["svm__C"]),
                                    "cv": {"folds": folds, "seed": seed, "data": "training pilot only",
                                           "results": results}}


def ordered_scores(pipeline, matrix, labels):
    """Decision scores reordered to the canonical label order."""
    raw = pipeline.decision_function(matrix)
    classes = list(pipeline.classes_)
    if len(classes) == 2:
        raise ValueError("Binary SVM decision scores are not supported for a 20-way pilot")
    scores = np.full((matrix.shape[0], len(labels)), -np.inf)
    for column, label in enumerate(classes):
        scores[:, labels.index(label)] = raw[:, column]
    return scores


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/manifests/train_pilot.jsonl")
    parser.add_argument("--train-metadata", type=Path,
                        default=ROOT / "data/manifests/train_pilot.metadata.json")
    parser.add_argument("--dev-manifest", type=Path, default=ROOT / "data/manifests/dev_pilot.jsonl")
    parser.add_argument("--dev-metadata", type=Path,
                        default=ROOT / "data/manifests/dev_pilot.metadata.json")
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Folder that the manifests' audio_path values are relative to")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=ROOT / "configs/labels.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for this run (must not exist)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Write predictions only; run the shared evaluator separately")
    return parser.parse_args(argv)


def run(args):
    started_at = utc_now()
    wall_start = time.perf_counter()
    config = load_config(args.config)
    labels = ev.load_labels(args.labels)
    seed = int(config["seed"])

    train_rows, train_split, train_meta = load_split(args.train_manifest, args.train_metadata,
                                                     args.labels, labels)
    dev_rows, dev_split, dev_meta = load_split(args.dev_manifest, args.dev_metadata,
                                               args.labels, labels)
    if train_split != "train" or dev_split != "dev":
        raise ValueError(f"Expected train and dev manifests, got {train_split} and {dev_split}")
    for field in ("dataset_version", "split_version"):
        if train_meta[field] != dev_meta[field]:
            raise ValueError(f"Train and dev metadata disagree on {field}")
    overlap = {r["clip_id"] for r in train_rows} & {r["clip_id"] for r in dev_rows}
    if overlap:
        raise ValueError(f"{len(overlap)} clip IDs appear in both train and dev")
    if not args.audio_root.is_dir():
        raise ValueError(f"Audio root not found: {args.audio_root}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    extractor = MfccExtractor(config["features"])
    timings = {}

    # 1. Training features and fit (training labels only).
    tick = time.perf_counter()
    train_x, train_ids, train_failures = extract_split(train_rows, args.audio_root, extractor, "train")
    timings["train_feature_extraction_sec"] = time.perf_counter() - tick
    label_by_id = {r["clip_id"]: r["label"] for r in train_rows}
    train_y = np.array([label_by_id[i] for i in train_ids])
    if len(set(train_y)) != len(labels):
        raise ValueError("Usable training features do not cover all 20 labels")

    tick = time.perf_counter()
    pipeline, selection = fit_classifier(train_x, train_y, config["classifier"], seed)
    timings["fit_including_cv_sec"] = time.perf_counter() - tick
    train_accuracy = float((pipeline.predict(train_x) == train_y).mean())

    # 2. Development features and predictions. Gold dev labels are stripped first.
    dev_inputs = without_labels(dev_rows)
    tick = time.perf_counter()
    dev_x, dev_ids, dev_failures = extract_split(dev_inputs, args.audio_root, extractor, "dev")
    timings["dev_feature_extraction_sec"] = time.perf_counter() - tick
    tick = time.perf_counter()
    dev_scores = ordered_scores(pipeline, dev_x, labels) if len(dev_ids) else np.empty((0, len(labels)))
    dev_predicted = pipeline.predict(dev_x) if len(dev_ids) else np.empty(0, dtype=str)
    timings["dev_predict_sec"] = time.perf_counter() - tick
    for row_scores, predicted in zip(dev_scores, dev_predicted):
        if labels[int(np.argmax(row_scores))] != predicted:
            raise RuntimeError("SVM prediction disagrees with argmax of its decision scores")

    # 3. Save the fitted model; its content hash is the immutable model revision.
    import joblib

    model_path = args.output_dir / "model.joblib"
    joblib.dump({"pipeline": pipeline, "labels": labels, "feature_names": feature_names(config["features"]),
                 "feature_config": config["features"], "result_role": RESULT_ROLE}, model_path)
    model_revision = f"{MODEL_NAME}:{ev.sha256_file(model_path)[:16]}"

    predictions = {cid: {"clip_id": cid, "predicted_label": str(lab), "model_revision": model_revision,
                         "status": "ok"} for cid, lab in zip(dev_ids, dev_predicted)}
    for failure in dev_failures:
        predictions[failure["clip_id"]] = {"clip_id": failure["clip_id"], "predicted_label": None,
                                           "model_revision": model_revision, "status": "audio_error",
                                           "error_code": failure["error_code"]}
    prediction_rows = [predictions[r["clip_id"]] for r in dev_rows]  # manifest order, every ID once
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    ev.validate_prediction_fields(ev.read_jsonl(predictions_path), labels)

    score_rows = [{"clip_id": cid, "score_labels": labels, "score_type": "svm_ovr_decision",
                   "scores": [float(x) for x in s]} for cid, s in zip(dev_ids, dev_scores)]
    write_jsonl(args.output_dir / "dev_decision_scores.jsonl", score_rows)
    np.savez_compressed(args.output_dir / "features.npz", train_x=train_x, train_ids=np.array(train_ids),
                        dev_x=dev_x, dev_ids=np.array(dev_ids),
                        feature_names=np.array(feature_names(config["features"])))
    write_jsonl(args.output_dir / "failures.jsonl",
                [{**f, "split": "train"} for f in train_failures] + [{**f, "split": "dev"} for f in dev_failures])
    write_json(args.output_dir / "config.json", config)

    # 4. Shared evaluator, exactly as other experiments run it.
    evaluation = None
    if not args.skip_evaluation:
        command = [sys.executable, str(ROOT / "src/evaluation/evaluate.py"),
                   "--manifest", str(args.dev_manifest), "--metadata", str(args.dev_metadata),
                   "--predictions", str(predictions_path), "--labels", str(args.labels),
                   "--split", "dev", "--model-name", MODEL_NAME,
                   "--output-dir", str(args.output_dir / "evaluation")]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Shared evaluator failed: {completed.stderr.strip()}")
        metrics = json.loads((args.output_dir / "evaluation/metrics.json").read_text(encoding="utf-8"))
        evaluation = {key: metrics[key] for key in
                      ("evaluation_mode", "expected", "successful", "failed", "missing", "correct",
                       "coverage", "accuracy", "macro_f1")}
        evaluation["command"] = shlex.join(command)

    finished_at = utc_now()
    timings["wall_time_sec"] = time.perf_counter() - wall_start
    record = {
        "task": "EEC-15 / W1-11",
        "result_role": RESULT_ROLE,
        "note": ("Pilot sanity reference that exercises the data/evaluation path; not a competitive "
                 "substitute for strong baselines. In-source ADI-20 development pilot, not out-of-domain."),
        "model_name": MODEL_NAME,
        "model_revision": model_revision,
        "status": "completed_with_clip_failures" if (train_failures or dev_failures) else "completed",
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "code": {**git_state(), "script_sha256": ev.sha256_file(Path(__file__)),
                 "config_sha256": ev.sha256_file(args.config)},
        "data": {"dataset_version": train_meta["dataset_version"],
                 "split_version": train_meta["split_version"],
                 "train_manifest_sha256": train_meta["manifest_sha256"],
                 "dev_manifest_sha256": dev_meta["manifest_sha256"],
                 "labels_sha256": train_meta["labels_sha256"],
                 "audio_root": str(args.audio_root)},
        "leakage_controls": {
            "scaler_fit_on": "training pilot features only (inside each CV fold and for the final refit)",
            "classifier_fit_on": "training pilot features and labels only",
            "hyperparameter_selection": "stratified CV within the training pilot; no development labels",
            "dev_labels_used_for": "shared evaluator only, after predictions were written",
        },
        "counts": {"train_clips": len(train_rows), "train_used": len(train_ids),
                   "train_failed": len(train_failures), "dev_clips": len(dev_rows),
                   "dev_predicted": len(dev_ids), "dev_failed": len(dev_failures)},
        "features": {"dim": extractor.dim, "code_version": config["features"]["code_version"]},
        "model_selection": selection,
        "train_accuracy_resubstitution": train_accuracy,
        "development_evaluation": evaluation,
        "resources": {**{k: round(v, 3) for k, v in timings.items()}, "device": "cpu",
                      "peak_rss_mib": peak_rss_mib()},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "processor": platform.processor() or None,
                        "packages": {name: version(name) for name in
                                     ("numpy", "scipy", "scikit-learn", "soundfile", "joblib")}},
        "artifacts": {name: ev.sha256_file(args.output_dir / name) for name in
                      ("model.joblib", "predictions.jsonl", "dev_decision_scores.jsonl",
                       "features.npz", "failures.jsonl", "config.json")},
    }
    write_json(args.output_dir / "run_record.json", record)

    progress(f"Selected C={selection['selected_C']}; training resubstitution accuracy {train_accuracy:.2%}")
    progress(f"Clip failures: train {len(train_failures)}, dev {len(dev_failures)}")
    progress(f"Wall time {timings['wall_time_sec']:.1f}s; peak RSS {record['resources']['peak_rss_mib']} MiB")
    progress(f"Run record: {args.output_dir / 'run_record.json'}")
    return 2 if (train_failures or dev_failures) else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"MFCC+SVM baseline error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
