#!/usr/bin/env python3
"""W1-12 / EEC-16: linear dialect head on cached frozen AraBEST-RQ features.

Pipeline
  1. Load the pinned encoder (frozen, eval mode) and fill one feature cache for the train
     and dev pilots. Clips already cached under the same provenance are reused, so train
     and dev features always come from one encoder revision and one feature config.
  2. Overfit gate: fit the head on a tiny training subset and stop unless it memorises it.
     Flat loss there points to broken features, misjoined labels or the optimiser, and is
     cheaper to find before the longer run.
  3. Train the 20-way linear head on the training pilot with the fixed configuration.
     Learning rate, epochs and the last-epoch checkpoint are fixed in advance; dev labels
     are only scored for the learning curve and never change training.
  4. Write all 20 dev logits in canonical label order, the learning curve and checkpoint,
     then score the predictions with the shared evaluator.

The head is the matched classifier reference for the verifier experiment
(docs/verifier_experiment.md); the dev pilot is in-source ADI-20, not out-of-domain.

Exit codes: 0 completed with no clip failures, 2 completed with explicit clip failures,
1 setup/contract error or failed overfit gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate as ev
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/acoustic_classifier.json"
MODEL_NAME = "arabest-rq-frozen-linear-pilot"
RESULT_ROLE = "pilot_frozen_encoder_reference_for_verifier"


# ---------------------------------------------------------------- configuration

def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    enc, head, opt, gate = config["encoder"], config["head"], config["optimization"], config["overfit_check"]
    if not enc["frozen"]:
        raise ValueError("Only a frozen encoder is implemented")
    if enc["checkpoint"] != ae.REPO_ID or enc["revision"] != ae.REVISION:
        raise ValueError("Only the pinned AraBEST-RQ-300M-6k revision is supported")
    if enc["feature_code_version"] != ae.FEATURE_CODE_VERSION:
        raise ValueError(f"Config expects features {enc['feature_code_version']}, "
                         f"code produces {ae.FEATURE_CODE_VERSION}")
    if (head["loss"], head["label_smoothing"], head["weight_init"], head["bias_init"]) != (
            "cross_entropy", 0.0, "xavier_uniform", "zeros"):
        raise ValueError("Only an unsmoothed cross-entropy head with xavier/zero init is implemented")
    if head["device"] != "cpu":
        raise ValueError("The head is trained on CPU so runs are reproducible")
    if (opt["optimizer"] != "AdamW" or opt["scheduler"] is not None or opt["early_stopping"]
            or opt["checkpoint_selection"] != "last_epoch" or opt["drop_last"]
            or opt["augmentation"] or opt["class_rebalancing"]):
        raise ValueError("Only AdamW without scheduler, early stopping, augmentation or "
                         "rebalancing, keeping the last epoch, is implemented")
    for settings in (opt, gate):
        if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["learning_rate"] <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
    if gate["clips_per_label"] < 1:
        raise ValueError("overfit_check.clips_per_label must be positive")
    return config


def feature_config(encoder_config):
    layers = encoder_config["cached_layers"]
    return ae.FeatureConfig(layers=layers if layers == "all" else tuple(layers),
                            pooling=encoder_config["pooling"], precision=encoder_config["precision"],
                            max_duration_sec=encoder_config["max_duration_sec"])


# ---------------------------------------------------------------- encoder and features

def state_sha256(module):
    """Fingerprint of every parameter and buffer, to show extraction left them unchanged."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_summary(encoder):
    params = list(encoder.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    if trainable or encoder.training:
        raise RuntimeError(f"Encoder is not frozen: {trainable} trainable parameters, "
                           f"training mode {encoder.training}")
    return {"parameters": sum(p.numel() for p in params), "trainable_parameters": trainable,
            "training_mode": encoder.training}


def prepare_features(encoder, cache, rows, audio_root, settings, split_name):
    """Extract and cache clips that are not cached yet. Returns (extracted_ids, failures).

    Rows carry audio identity only. Each WAV is checked against the manifest before
    decoding; a clip that fails becomes an explicit failure, never a silent exclusion.
    """
    todo = cache.missing(rows)
    if todo and audio_root is None:
        raise ValueError(f"[{split_name}] {len(todo)} clips are not cached; --audio-root is required")
    ready, failures = [], []
    for row in todo:
        try:
            base.verify_audio(ae.resolve_audio_path(audio_root, row["audio_path"]), row)
            ready.append(row)
        except Exception as error:  # noqa: BLE001 - every clip failure is recorded explicitly
            failures.append({"clip_id": row["clip_id"],
                             "error_code": ("audio_integrity_mismatch" if isinstance(error, base.AudioIntegrityError)
                                            else "audio_unreadable"),
                             "error_type": type(error).__name__, "message": str(error)[:300]})
    started, done = time.perf_counter(), 0
    for result in encoder.iter_extract_manifest(ready, audio_root, batch_size=settings["batch_size"],
                                               chunk_size=settings["chunk_size"]):
        cache.write(result)
        done += len(result.clip_ids)
        base.progress(f"[{split_name}] features {done}/{len(ready)} ({time.perf_counter() - started:.1f}s)")
    base.progress(f"[{split_name}] {len(rows) - len(todo)} clips reused from cache, {len(ready)} extracted, "
                  f"{len(failures)} failed")
    return [row["clip_id"] for row in ready], failures


def cache_keys_sha256(cache, clip_ids):
    """Hash of the exact cache entries a split reads, in order."""
    index = cache.index()
    return base.sha256_text(base.canonical_json([[c, index[c]["cache_key"]] for c in clip_ids]))


def load_layer(cache, clip_ids, position):
    return np.ascontiguousarray(cache.load(clip_ids).features[:, position, :], dtype=np.float32)


def feature_stats(matrix):
    """Cheap signs of broken features: nonfinite values, constant dimensions, collapsed clips."""
    if not len(matrix):
        return None
    unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    cosine = unit @ unit.T
    off_diagonal = cosine[~np.eye(len(matrix), dtype=bool)]
    return {"clips": len(matrix), "finite": bool(np.isfinite(matrix).all()),
            "mean": float(matrix.mean()), "mean_abs": float(np.abs(matrix).mean()),
            "max_abs": float(np.abs(matrix).max()), "mean_dim_std": float(matrix.std(0).mean()),
            "constant_dims": int((matrix.std(0) == 0).sum()),
            "duplicate_rows": int(len(matrix) - len(np.unique(matrix, axis=0))),
            "mean_offdiag_cosine": float(off_diagonal.mean()) if off_diagonal.size else None}


# ---------------------------------------------------------------- head

def run_seeds(seed):
    """Independent initialisation and shuffling seeds derived from the run seed."""
    init_seed, shuffle_seed = np.random.SeedSequence(seed).generate_state(2)
    return int(init_seed), int(shuffle_seed)


def build_head(dim, num_labels, seed):
    head = torch.nn.Linear(dim, num_labels)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(head.weight, generator=torch.Generator().manual_seed(seed))
        head.bias.zero_()
    return head


@torch.no_grad()
def score_split(head, x, y):
    head.eval()
    logits = head(x)
    return {"loss": float(F.cross_entropy(logits, y)),
            "accuracy": float((logits.argmax(1) == y).float().mean())}


def train_head(train_x, train_y, settings, seed, monitor=None):
    """Fit the linear head on (train_x, train_y). Returns (head, curve).

    monitor maps extra split names to (x, y) scored after every epoch for the curve only;
    no training step reads it.
    """
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    splits = {"train": (x, y), **{name: (torch.from_numpy(mx), torch.from_numpy(my))
                                  for name, (mx, my) in (monitor or {}).items() if len(mx)}}
    init_seed, shuffle_seed = run_seeds(seed)
    head = build_head(x.shape[1], settings["num_labels"], init_seed)
    optimizer = torch.optim.AdamW(
        [{"params": [head.weight], "weight_decay": settings["weight_decay_W"]},
         {"params": [head.bias], "weight_decay": settings["weight_decay_bias"]}],
        lr=settings["learning_rate"], betas=tuple(settings["betas"]), eps=settings["eps"])
    shuffle = torch.Generator().manual_seed(shuffle_seed)
    updates = 0
    curve = [{"epoch": 0, "updates": 0, **{n: score_split(head, *d) for n, d in splits.items()}}]
    for epoch in range(1, settings["epochs"] + 1):
        head.train()
        batch_losses, grad_norms = [], []
        order = torch.randperm(len(x), generator=shuffle)
        for start in range(0, len(x), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            loss = F.cross_entropy(head(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(head.parameters(),
                                                                   settings["gradient_clip_norm"])))
            optimizer.step()
            batch_losses.append(loss.item())
            updates += 1
        curve.append({"epoch": epoch, "updates": updates, "mean_batch_loss": float(np.mean(batch_losses)),
                      "max_grad_norm_before_clip": max(grad_norms),
                      **{n: score_split(head, *d) for n, d in splits.items()}})
    head.eval()
    return head, curve


def overfit_subset(train_ids, label_by_id, labels, per_label):
    """First per_label usable training clips of every label, in manifest order."""
    chosen = {label: [] for label in labels}
    for clip_id in train_ids:
        bucket = chosen[label_by_id[clip_id]]
        if len(bucket) < per_label:
            bucket.append(clip_id)
    if any(len(bucket) < per_label for bucket in chosen.values()):
        raise ValueError(f"Overfit check needs {per_label} usable training clips for every label")
    return [clip_id for label in labels for clip_id in chosen[label]]


def overfit_check(train_x, train_y, train_ids, label_by_id, labels, config):
    gate, seed = config["overfit_check"], int(config["seed"])
    subset = overfit_subset(train_ids, label_by_id, labels, gate["clips_per_label"])
    rows = [train_ids.index(clip_id) for clip_id in subset]
    settings = {**config["optimization"], "num_labels": len(labels),
                **{k: gate[k] for k in ("epochs", "batch_size", "learning_rate")}}
    _, curve = train_head(train_x[rows], train_y[rows], settings, seed)
    initial, final = curve[0]["train"], curve[-1]["train"]
    passed = (final["accuracy"] >= gate["min_train_accuracy"]
              and final["loss"] <= gate["max_final_loss_ratio"] * initial["loss"])
    return {"passed": passed, "clip_ids": subset, "settings": gate, "chance_loss": math.log(len(labels)),
            "initial_loss": initial["loss"], "final_loss": final["loss"],
            "initial_accuracy": initial["accuracy"], "final_accuracy": final["accuracy"],
            "feature_stats": feature_stats(train_x[rows]), "curve": curve}


def predict(head, matrix):
    """All 20 logits in canonical order; ties go to the lowest canonical index."""
    with torch.no_grad():
        logits = head(torch.from_numpy(matrix)).numpy().astype(np.float64)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    return logits, np.argmax(logits, axis=1)


def run_evaluator(args, predictions_path, output_dir):
    command = [sys.executable, str(ROOT / "src/evaluation/evaluate.py"),
               "--manifest", str(args.dev_manifest), "--metadata", str(args.dev_metadata),
               "--predictions", str(predictions_path), "--labels", str(args.labels),
               "--split", "dev", "--model-name", MODEL_NAME, "--output-dir", str(output_dir)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Shared evaluator failed: {completed.stderr.strip()}")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    evaluation = {key: metrics[key] for key in
                  ("evaluation_mode", "expected", "successful", "failed", "missing", "correct",
                   "coverage", "accuracy", "macro_f1")}
    evaluation["command"] = shlex.join(command)
    return evaluation


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
    parser.add_argument("--audio-root", type=Path,
                        help="Folder the manifests' audio_path values are relative to; "
                             "only needed for clips that are not cached yet")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"),
                        help="Root of the reusable feature cache (default: $OUTPUT_ROOT)")
    parser.add_argument("--cache-root", type=Path, default=os.environ.get("CACHE_ROOT"),
                        help="Hugging Face download cache (default: $CACHE_ROOT)")
    parser.add_argument("--device", help="Encoder device for extraction (default: cuda if available)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=ROOT / "configs/labels.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for this run (must not exist)")
    parser.add_argument("--extract-batch-size", type=int,
                        help="Encoder batch size for extraction (default: from the config); "
                             "does not change feature values beyond the documented tolerance")
    parser.add_argument("--overfit-check-only", action="store_true",
                        help="Extract features and run the overfit gate, then stop before pilot training")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Write predictions only; run the shared evaluator separately")
    args = parser.parse_args(argv)
    if args.extract_batch_size is not None and args.extract_batch_size < 1:
        parser.error("--extract-batch-size must be positive")
    if args.output_root is None:
        parser.error("--output-root or OUTPUT_ROOT is required")
    return args


def run(args, load_encoder=ae.AcousticEncoder.from_pretrained):
    started_at = base.utc_now()
    wall_start = time.perf_counter()
    config = load_config(args.config)
    labels = ev.load_labels(args.labels)
    seed = int(config["seed"])

    train_rows, train_split, train_meta = base.load_split(args.train_manifest, args.train_metadata,
                                                          args.labels, labels)
    dev_rows, dev_split, dev_meta = base.load_split(args.dev_manifest, args.dev_metadata,
                                                    args.labels, labels)
    if train_split != "train" or dev_split != "dev":
        raise ValueError(f"Expected train and dev manifests, got {train_split} and {dev_split}")
    for field in ("dataset_version", "split_version"):
        if train_meta[field] != dev_meta[field]:
            raise ValueError(f"Train and dev metadata disagree on {field}")
    overlap = {r["clip_id"] for r in train_rows} & {r["clip_id"] for r in dev_rows}
    if overlap:
        raise ValueError(f"{len(overlap)} clip IDs appear in both train and dev")
    if args.audio_root is not None and not args.audio_root.is_dir():
        raise ValueError(f"Audio root not found: {args.audio_root}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    timings = {}
    extraction = {**config["encoder"], "batch_size": args.extract_batch_size or config["encoder"]["batch_size"]}

    # 1. One frozen encoder and one cache serve both splits.
    tick = time.perf_counter()
    encoder = load_encoder(feature_config(config["encoder"]), device=args.device, cache_dir=args.cache_root)
    provenance = encoder.provenance()
    if provenance["feature_config"]["checkpoint_revision"] != config["encoder"]["revision"]:
        raise ValueError("Encoder revision differs from the config")
    if provenance["checkpoint"]["revision"] not in (None, config["encoder"]["revision"]):
        raise ValueError("Loaded checkpoint revision differs from the config")
    if provenance["output_dim"] != config["encoder"]["feature_dim"]:
        raise ValueError(f"Encoder output dim {provenance['output_dim']} differs from the config")
    if config["encoder"]["layer_name"] not in provenance["layer_names"]:
        raise ValueError(f"Layer {config['encoder']['layer_name']} is not in the cached layers")
    frozen = frozen_summary(encoder)
    state_before = state_sha256(encoder)
    encoder_device = encoder.device
    timings["encoder_load_sec"] = time.perf_counter() - tick
    cache = ae.FeatureCache(args.output_root, provenance)

    if encoder_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(encoder_device)
    tick = time.perf_counter()
    train_extracted, train_failures = prepare_features(encoder, cache, base.without_labels(train_rows),
                                                       args.audio_root, extraction, "train")
    dev_extracted, dev_failures = prepare_features(encoder, cache, base.without_labels(dev_rows),
                                                   args.audio_root, extraction, "dev")
    timings["feature_extraction_sec"] = time.perf_counter() - tick
    duration_by_id = {r["clip_id"]: r.get("duration_sec") or 0.0 for r in train_rows + dev_rows}
    extracted_audio_sec = sum(duration_by_id[c] for c in train_extracted + dev_extracted)
    peak_gpu = (torch.cuda.max_memory_allocated(encoder_device) if encoder_device.type == "cuda" else None)
    frozen.update(state_sha256_before=state_before, state_sha256_after=state_sha256(encoder))
    if frozen["state_sha256_before"] != frozen["state_sha256_after"]:
        raise RuntimeError("Encoder parameters changed during feature extraction")
    del encoder

    # The head runs on CPU with one thread so a rerun reproduces the same checkpoint.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    failed = {f["clip_id"] for f in train_failures + dev_failures}
    train_ids = [r["clip_id"] for r in train_rows if r["clip_id"] not in failed]
    dev_ids = [r["clip_id"] for r in dev_rows if r["clip_id"] not in failed]
    position = provenance["layer_names"].index(config["encoder"]["layer_name"])
    train_x, dev_x = load_layer(cache, train_ids, position), load_layer(cache, dev_ids, position)
    label_by_id = {r["clip_id"]: r["label"] for r in train_rows}
    train_y = np.array([labels.index(label_by_id[c]) for c in train_ids], dtype=np.int64)
    if len(set(train_y.tolist())) != len(labels):
        raise ValueError("Usable training features do not cover all 20 labels")
    base.write_jsonl(args.output_dir / "failures.jsonl",
                     [{**f, "split": "train"} for f in train_failures] + [{**f, "split": "dev"} for f in dev_failures])
    base.write_json(args.output_dir / "config.json", config)

    record = {
        "task": "EEC-16 / W1-12",
        "result_role": RESULT_ROLE,
        "note": ("Linear head on frozen AraBEST-RQ features; matched classifier reference for the "
                 "verifier experiment. In-source ADI-20 development pilot, not out-of-domain."),
        "model_name": MODEL_NAME,
        "status": None,
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at_utc": started_at.isoformat(),
        "code": {**base.git_state(), "script_sha256": ev.sha256_file(Path(__file__)),
                 "encoder_code_sha256": ev.sha256_file(Path(ae.__file__)),
                 "config_sha256": ev.sha256_file(args.config)},
        "data": {"dataset_version": train_meta["dataset_version"],
                 "split_version": train_meta["split_version"],
                 "train_manifest_sha256": train_meta["manifest_sha256"],
                 "dev_manifest_sha256": dev_meta["manifest_sha256"],
                 "labels_sha256": train_meta["labels_sha256"],
                 "audio_root": str(args.audio_root) if args.audio_root else None},
        "encoder": {"repo_id": provenance["checkpoint"]["repo_id"],
                    "revision": provenance["feature_config"]["checkpoint_revision"],
                    "checkpoint_files_sha256": provenance["checkpoint"]["files"],
                    "feature_code_version": provenance["feature_config"]["code_version"],
                    "frozen": frozen,
                    "head_input": "cached numpy features; no autograd path reaches the encoder"},
        "features": {"cache_root": str(cache.root),
                     "feature_config_sha256": provenance["feature_config_sha256"],
                     "provenance_sha256": base.sha256_text(base.canonical_json(provenance)),
                     "single_cache_for_train_and_dev": True,
                     "layer_name": config["encoder"]["layer_name"],
                     "layer_index": provenance["layer_indices"][position],
                     "dim": provenance["output_dim"],
                     "extraction_batch_size": extraction["batch_size"],
                     "train_cache_keys_sha256": cache_keys_sha256(cache, train_ids),
                     "dev_cache_keys_sha256": cache_keys_sha256(cache, dev_ids),
                     "extracted_this_run": {"train": len(train_extracted), "dev": len(dev_extracted)},
                     "extracted_audio_sec": round(extracted_audio_sec, 3),
                     "reused_from_cache": {"train": len(train_ids) - len(train_extracted),
                                           "dev": len(dev_ids) - len(dev_extracted)},
                     "train_stats": feature_stats(train_x), "dev_stats": feature_stats(dev_x)},
        "leakage_controls": {
            "head_fit_on": "training pilot features and labels only",
            "hyperparameters": "fixed in the config before training; no search",
            "checkpoint_selection": "last epoch; no early stopping",
            "dev_labels_used_for": "learning-curve diagnostics and the shared evaluator; never for training decisions",
        },
        "counts": {"train_clips": len(train_rows), "train_used": len(train_ids),
                   "train_failed": len(train_failures), "dev_clips": len(dev_rows),
                   "dev_used": len(dev_ids), "dev_failed": len(dev_failures)},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "processor": platform.processor() or None,
                        "gpu": torch.cuda.get_device_name(encoder_device) if encoder_device.type == "cuda" else None,
                        "packages": {name: version(name) for name in
                                     ("numpy", "torch", "torchaudio", "speechbrain", "soundfile")}},
    }

    def finish(status, artifacts):
        timings["wall_time_sec"] = time.perf_counter() - wall_start
        record.update(status=status, finished_at_utc=base.utc_now().isoformat())
        record["resources"] = {**{k: round(v, 3) for k, v in timings.items()},
                               "extraction_audio_sec_per_sec": (round(extracted_audio_sec / timings["feature_extraction_sec"], 2)
                                                                if extracted_audio_sec else None),
                               "encoder_device": str(encoder_device), "head_device": "cpu",
                               "peak_gpu_memory_allocated_bytes": peak_gpu, "peak_rss_mib": base.peak_rss_mib()}
        record["artifacts"] = {name: ev.sha256_file(args.output_dir / name) for name in artifacts}
        base.write_json(args.output_dir / "run_record.json", record)
        base.progress(f"Clip failures: train {len(train_failures)}, dev {len(dev_failures)}")
        base.progress(f"Wall time {timings['wall_time_sec']:.1f}s; peak RSS {record['resources']['peak_rss_mib']} MiB")
        base.progress(f"Run record: {args.output_dir / 'run_record.json'}")

    # 2. Overfit gate on a tiny training subset before the longer run.
    tick = time.perf_counter()
    gate = overfit_check(train_x, train_y, train_ids, label_by_id, labels, config)
    timings["overfit_check_sec"] = time.perf_counter() - tick
    base.write_json(args.output_dir / "overfit_check.json", gate)
    record["overfit_check"] = {k: gate[k] for k in ("passed", "initial_loss", "final_loss",
                                                    "initial_accuracy", "final_accuracy", "chance_loss")}
    base.progress(f"Overfit check on {len(gate['clip_ids'])} clips: loss {gate['initial_loss']:.4f} -> "
                  f"{gate['final_loss']:.4f}, accuracy {gate['final_accuracy']:.2%} "
                  f"({'passed' if gate['passed'] else 'FAILED'})")
    gate_artifacts = ["overfit_check.json", "failures.jsonl", "config.json"]
    if not gate["passed"]:
        finish("overfit_check_failed", gate_artifacts)
        raise RuntimeError("Overfit check failed; inspect overfit_check.json before training longer")
    if args.overfit_check_only:
        finish("overfit_check_only", gate_artifacts)
        return 2 if (train_failures or dev_failures) else 0

    # 3. Pilot training. Dev labels feed the learning curve only.
    dev_gold = {r["clip_id"]: r["label"] for r in dev_rows}
    dev_y = np.array([labels.index(dev_gold[c]) for c in dev_ids], dtype=np.int64)
    settings = {**config["optimization"], "num_labels": len(labels)}
    tick = time.perf_counter()
    head, curve = train_head(train_x, train_y, settings, seed, monitor={"dev": (dev_x, dev_y)})
    timings["train_sec"] = time.perf_counter() - tick
    base.write_jsonl(args.output_dir / "learning_curve.jsonl", curve)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save({"state_dict": head.state_dict(), "labels": labels,
                "layer_name": config["encoder"]["layer_name"],
                "layer_index": provenance["layer_indices"][position],
                "feature_dim": provenance["output_dim"],
                "feature_provenance_sha256": record["features"]["provenance_sha256"],
                "config": config, "result_role": RESULT_ROLE}, checkpoint_path)
    model_revision = f"{MODEL_NAME}:{ev.sha256_file(checkpoint_path)[:16]}"

    # 4. Dev predictions with full class scores, in manifest order.
    tick = time.perf_counter()
    logits, predicted = predict(head, dev_x)
    timings["dev_predict_sec"] = time.perf_counter() - tick
    predictions = {cid: {"clip_id": cid, "predicted_label": labels[int(p)], "model_revision": model_revision,
                         "status": "ok", "scores": [float(v) for v in row], "score_labels": labels,
                         "score_type": "logits"}
                   for cid, p, row in zip(dev_ids, predicted, logits)}
    for failure in dev_failures:
        predictions[failure["clip_id"]] = {"clip_id": failure["clip_id"], "predicted_label": None,
                                           "model_revision": model_revision, "status": "audio_error",
                                           "error_code": failure["error_code"]}
    prediction_rows = [predictions[r["clip_id"]] for r in dev_rows]
    predictions_path = args.output_dir / "predictions.jsonl"
    base.write_jsonl(predictions_path, prediction_rows)
    ev.validate_prediction_fields(ev.read_jsonl(predictions_path), labels)

    evaluation = None
    if not args.skip_evaluation:
        evaluation = run_evaluator(args, predictions_path, args.output_dir / "evaluation")

    final = curve[-1]
    record.update({
        "model_revision": model_revision,
        "head": {"trainable_parameters": sum(p.numel() for p in head.parameters()),
                 "seeds": dict(zip(("init", "shuffle"), run_seeds(seed)), run=seed),
                 "updates": final["updates"]},
        "final_epoch": {"epoch": final["epoch"], "train": final["train"], "dev": final.get("dev")},
        "development_evaluation": evaluation,
    })
    base.progress(f"Final epoch {final['epoch']}: train loss {final['train']['loss']:.4f} "
                  f"accuracy {final['train']['accuracy']:.2%}"
                  + (f"; dev loss {final['dev']['loss']:.4f} accuracy {final['dev']['accuracy']:.2%}"
                     if "dev" in final else ""))
    finish("completed_with_clip_failures" if (train_failures or dev_failures) else "completed",
           [*gate_artifacts, "checkpoint.pt", "predictions.jsonl", "learning_curve.jsonl"])
    return 2 if (train_failures or dev_failures) else 0


def main(argv=None, load_encoder=ae.AcousticEncoder.from_pretrained):
    args = parse_args(argv)
    try:
        return run(args, load_encoder)
    except (OSError, ValueError, RuntimeError, KeyError, FloatingPointError) as error:
        print(f"Acoustic classifier error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
