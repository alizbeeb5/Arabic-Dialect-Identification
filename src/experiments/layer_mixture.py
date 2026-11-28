#!/usr/bin/env python3
"""Idea 1 pilot: layer sweep and learned layer mixture on cached frozen AraBEST-RQ features.

Reads the 26-layer mean-pooled features that src/experiments/acoustic_classifier.py already
cached for the frozen pilot. It never loads the encoder or any audio, and it stops if a manifest
clip is missing from that cache or was cached from different audio.

Arms. Every head is a 20-way linear layer trained on CPU with the W1-12 optimiser. Every arm
except b0_final_raw reads per-layer z-scored features (statistics from the fit's training rows).
  b0_final        final layer (index 25): the W1-12 head with a chosen number of epochs
  b0_final_raw    the same without z-scoring, as W1-12 fed it
  layer_<i>       one probe per cached layer (the sweep); the final layer's probe is b0_final
  selected_layer  the sweep layer with the best training-CV accuracy
  mixture_<name>  learned softmax weights over a set of layers from the config, then the head

Selection uses training data only. Grouped, label-stratified K-fold CV on the training pilot
picks each arm's number of epochs from a fixed grid and picks the selected layer. Dev labels are
read only after every choice is written to selection.json, to score the final fits.

These are in-source ADI-20 development results (on the 200-clip pilot one clip is 0.5 points).
They decide what to try next; they are not out-of-domain or benchmark results.

Exit codes: 0 completed, 1 setup/contract error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import freeze_splits as fs
from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/layer_mixture.json"
MODEL_NAME = "arabest-rq-frozen-layer-mixture-pilot"
BASELINE, BASELINE_RAW, SELECTED = "b0_final", "b0_final_raw", "selected_layer"


# ---------------------------------------------------------------- configuration

def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    enc, head, opt, sel = config["encoder"], config["head"], config["optimization"], config["selection"]
    if not enc["frozen"] or enc["cached_layers"] != "all":
        raise ValueError("The pilot reads a frozen encoder's cache that stores all layers")
    if enc["checkpoint"] != ae.REPO_ID or enc["revision"] != ae.REVISION:
        raise ValueError("Only the pinned AraBEST-RQ-300M-6k revision is supported")
    if enc["feature_code_version"] != ae.FEATURE_CODE_VERSION:
        raise ValueError(f"Config expects features {enc['feature_code_version']}, "
                         f"code produces {ae.FEATURE_CODE_VERSION}")
    if (head["loss"], head["label_smoothing"], head["weight_init"], head["bias_init"], head["device"]) != (
            "cross_entropy", 0.0, "xavier_uniform", "zeros", "cpu"):
        raise ValueError("Only an unsmoothed cross-entropy CPU head with xavier/zero init is implemented")
    if (opt["optimizer"] != "AdamW" or opt["scheduler"] is not None or opt["drop_last"]
            or opt["layer_weight_init"] != "uniform"):
        raise ValueError("Only AdamW without scheduler, keeping partial batches, with uniform "
                         "initial layer weights is implemented")
    if config["preprocessing"]["standardize"] != "per_layer_per_dim_zscore":
        raise ValueError("Only per-layer, per-dimension z-scoring is implemented")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    epochs = sel["candidate_epochs"]
    if not epochs or epochs != sorted(set(epochs)) or epochs[0] < 1:
        raise ValueError("selection.candidate_epochs must be increasing positive integers")
    if sel["cv_folds"] < 2:
        raise ValueError("selection.cv_folds must be at least 2")
    if any(not name.startswith("mixture_") for name in config["mixtures"]):
        raise ValueError("Mixture arm names must start with 'mixture_'")
    return config


def expected_provenance(feature_config, architecture=ae.Architecture(), checkpoint=None):
    """The provenance AcousticEncoder.provenance() reports, built without loading the model.

    FeatureCache checks it against the stored config.json and every cache key, so a cache built
    from another checkpoint, feature config or code version stops the run.
    """
    indices = feature_config.layer_indices(architecture)
    return {"encoder": feature_config.encoder,
            "checkpoint": checkpoint if checkpoint is not None else {
                "repo_id": ae.REPO_ID, "revision": feature_config.checkpoint_revision,
                "files": dict(ae.CHECKPOINT_SHA256)},
            "architecture": asdict(architecture), "feature_config": feature_config.to_dict(),
            "feature_config_sha256": feature_config.sha256(), "layer_indices": list(indices),
            "layer_names": ae.layer_names(architecture, indices),
            "output_dim": architecture.d_model * (2 if feature_config.pooling == "mean_std" else 1),
            "frame_shift_sec": ae.DOWNSAMPLE * feature_config.hop_length_ms / 1000}


def arm_specs(config, layer_names):
    """Layers and preprocessing of every trained arm. Aliases are added after selection."""
    count = len(layer_names)
    if config["baseline_layer"] not in layer_names:
        raise ValueError(f"Baseline layer {config['baseline_layer']} is not cached")
    final = layer_names.index(config["baseline_layer"])
    specs = {f"layer_{i}": {"layers": [i], "standardize": True, "mixture": False} for i in range(count)}
    specs[BASELINE_RAW] = {"layers": [final], "standardize": False, "mixture": False}
    for name, layers in config["mixtures"].items():
        indices = list(range(count)) if layers == "all" else list(layers)
        if len(indices) < 2 or len(set(indices)) != len(indices) or not all(0 <= i < count for i in indices):
            raise ValueError(f"{name}: needs at least two distinct layer indices in 0..{count - 1}")
        specs[name] = {"layers": indices, "standardize": True, "mixture": True}
    return specs, final


# ---------------------------------------------------------------- folds

def make_folds(rows, k, seed):
    """Grouped, label-stratified folds over the training rows. Returns (fold per row, group_of).

    Clips that share a recording ID or identical audio (the split-freeze grouping) stay in one
    fold. Within a label, larger groups are placed first, each into the fold holding the fewest
    clips of that label; ties follow a seeded rotation so fold 0 is not always favoured.
    """
    groups, _ = fs.build_groups(rows)
    by_label, group_of = defaultdict(list), {}
    for gid, members in groups.items():
        labels = {m["label"] for m in members}
        if len(labels) != 1:
            raise ValueError(f"Group {gid} joins clips with different labels {sorted(labels)}")
        by_label[labels.pop()].append((gid, len(members)))
        group_of.update({m["clip_id"]: gid for m in members})
    fold_of_group = {}
    for label in sorted(by_label):
        counts = [0] * k
        offset = int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest(), 16) % k
        ordered = sorted(by_label[label],
                         key=lambda g: (-g[1], hashlib.sha256(f"{seed}:{g[0]}".encode()).hexdigest()))
        for gid, size in ordered:
            fold = min(range(k), key=lambda f: (counts[f], (f - offset) % k))
            fold_of_group[gid] = fold
            counts[fold] += size
    return np.array([fold_of_group[group_of[r["clip_id"]]] for r in rows], dtype=np.int64), group_of


# ---------------------------------------------------------------- heads

class LayerMixture(torch.nn.Module):
    """Softmax-weighted sum over pooled layers, then the linear head."""

    def __init__(self, n_layers, linear):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_layers))  # uniform weights at init
        self.linear = linear

    def layer_weights(self):
        return self.layer_logits.softmax(0)

    def forward(self, x):  # [B, L, D] -> [B, C]
        return self.linear((x * self.layer_weights()[None, :, None]).sum(1))


def build_model(x_shape, num_labels, init_seed):
    linear = clf.build_head(x_shape[-1], num_labels, init_seed)
    return linear if len(x_shape) == 2 else LayerMixture(x_shape[1], linear)


def make_optimizer(model, opt):
    linear = model.linear if isinstance(model, LayerMixture) else model
    groups = [{"params": [linear.weight], "weight_decay": opt["weight_decay_W"]},
              {"params": [linear.bias], "weight_decay": opt["weight_decay_bias"]}]
    if isinstance(model, LayerMixture):
        groups.append({"params": [model.layer_logits], "weight_decay": opt["layer_weight_decay"],
                       "lr": opt["layer_weight_learning_rate"]})
    return torch.optim.AdamW(groups, lr=opt["learning_rate"], betas=tuple(opt["betas"]), eps=opt["eps"])


def fit(x, y, opt, num_labels, seed, epochs, on_epoch=None):
    """Train for max(epochs) epochs; on_epoch(model, epoch) runs after every listed epoch.

    There is no scheduler and the seeds are fixed, so the model after e epochs of a longer run
    equals a run of exactly e epochs: one run scores every candidate budget.
    """
    init_seed, shuffle_seed = clf.run_seeds(seed)
    model = build_model(tuple(x.shape), num_labels, init_seed)
    optimizer = make_optimizer(model, opt)
    shuffle = torch.Generator().manual_seed(shuffle_seed)
    checkpoints = set(epochs)
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = torch.randperm(len(x), generator=shuffle)
        for start in range(0, len(x), opt["batch_size"]):
            batch = order[start:start + opt["batch_size"]]
            loss = F.cross_entropy(model(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["gradient_clip_norm"])
            optimizer.step()
        if on_epoch is not None and epoch in checkpoints:
            model.eval()
            on_epoch(model, epoch)
    model.eval()
    return model


@torch.no_grad()
def logits_of(model, x):
    logits = model(x).numpy().astype(np.float64)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    return logits


def classification_metrics(y_true, y_pred, num_labels):
    """Accuracy and macro-F1 over labels with reference support (the protocol's convention)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for c in range(num_labels):
        support = int((y_true == c).sum())
        if support:
            tp = int(((y_true == c) & (y_pred == c)).sum())
            fp = int(((y_true != c) & (y_pred == c)).sum())
            f1.append(2 * tp / (2 * tp + fp + support - tp))
    return {"accuracy": float((y_true == y_pred).mean()), "macro_f1": float(np.mean(f1))}


def zscore(fit_rows, others, floor):
    """Per-layer, per-dimension z-scores with statistics from fit_rows only."""
    mean, std = fit_rows.mean(0), np.maximum(fit_rows.std(0), floor)
    return [((m - mean) / std).astype(np.float32) for m in (fit_rows, *others)]


def arm_input(standardized, raw, spec):
    source = standardized if spec["standardize"] else raw
    picked = source[:, spec["layers"], :] if spec["mixture"] else source[:, spec["layers"][0], :]
    return torch.from_numpy(np.ascontiguousarray(picked, dtype=np.float32))


def paired(arm_correct, baseline_correct):
    """Paired dev comparison with an exact two-sided McNemar (binomial) test."""
    a, b = np.asarray(arm_correct, bool), np.asarray(baseline_correct, bool)
    only_arm, only_base = int((a & ~b).sum()), int((~a & b).sum())
    n = only_arm + only_base
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_arm, only_base) + 1)) / 2 ** n)
    return {"both_correct": int((a & b).sum()), "only_arm_correct": only_arm,
            "only_baseline_correct": only_base, "both_wrong": int((~a & ~b).sum()), "exact_mcnemar_p": p}


# ---------------------------------------------------------------- selection

def cross_validate(train_x, train_y, folds, specs, config, log):
    sel, opt = config["selection"], config["optimization"]
    num_labels, floor = config["head"]["num_labels"], config["preprocessing"]["std_floor"]
    rows = []
    for fold in range(sel["cv_folds"]):
        fit_mask, held = folds != fold, folds == fold
        if not held.any():
            raise ValueError(f"CV fold {fold} received no training clips")
        std_fit, std_held = zscore(train_x[fit_mask], [train_x[held]], floor)
        y_fit, y_held = torch.from_numpy(train_y[fit_mask]), train_y[held]
        tick = time.perf_counter()
        for name, spec in specs.items():
            x_fit = arm_input(std_fit, train_x[fit_mask], spec)
            x_held = arm_input(std_held, train_x[held], spec)
            for seed in config["seeds"]:
                def score(model, epoch, name=name, seed=seed):
                    predicted = logits_of(model, x_held).argmax(1)
                    rows.append({"arm": name, "fold": fold, "seed": seed, "epochs": epoch,
                                 "held_out_clips": int(held.sum()),
                                 **classification_metrics(y_held, predicted, num_labels)})
                fit(x_fit, y_fit, opt, num_labels, seed, sel["candidate_epochs"], score)
        log(f"CV fold {fold + 1}/{sel['cv_folds']}: {int(held.sum())} held-out clips, "
            f"{len(specs)} arms x {len(config['seeds'])} seeds in {time.perf_counter() - tick:.1f}s")
    return rows


def select(cv_rows, specs, epochs_grid):
    """Epochs per arm, then the selected layer, from training-CV results only."""
    summary = {}
    for name in specs:
        by_epochs = {}
        for e in epochs_grid:
            hits = [r for r in cv_rows if r["arm"] == name and r["epochs"] == e]
            acc, f1 = [r["accuracy"] for r in hits], [r["macro_f1"] for r in hits]
            by_epochs[e] = {"cv_accuracy_mean": float(np.mean(acc)), "cv_accuracy_std": float(np.std(acc)),
                            "cv_macro_f1_mean": float(np.mean(f1)), "runs": len(hits)}
        best = max(epochs_grid, key=lambda e: (by_epochs[e]["cv_accuracy_mean"],
                                               by_epochs[e]["cv_macro_f1_mean"], -e))
        summary[name] = {"epochs": best, **by_epochs[best],
                         "by_epochs": {str(e): v for e, v in by_epochs.items()}}
    layer_arms = [name for name in specs if name.startswith("layer_")]
    selected = max(layer_arms, key=lambda a: (summary[a]["cv_accuracy_mean"], summary[a]["cv_macro_f1_mean"],
                                              -int(a.split("_")[1])))
    return summary, selected


# ---------------------------------------------------------------- outputs

def run_evaluator(args, predictions_path, output_dir, model_name):
    command = [sys.executable, str(ROOT / "src/evaluation/evaluate.py"),
               "--manifest", str(args.dev_manifest), "--metadata", str(args.dev_metadata),
               "--predictions", str(predictions_path), "--labels", str(args.labels),
               "--split", "dev", "--model-name", model_name, "--output-dir", str(output_dir)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Shared evaluator failed: {completed.stderr.strip()}")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return {**{k: metrics[k] for k in ("expected", "correct", "coverage", "accuracy", "macro_f1")},
            "command": shlex.join(command)}


def write_predictions(path, dev_rows, logits, labels, model_revision):
    rows = [{"clip_id": r["clip_id"], "predicted_label": labels[int(np.argmax(row))],
             "model_revision": model_revision, "status": "ok", "scores": [float(v) for v in row],
             "score_labels": labels, "score_type": "logits"} for r, row in zip(dev_rows, logits)]
    base.write_jsonl(path, rows)
    ev.validate_prediction_fields(ev.read_jsonl(path), labels)


def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": [float(v) for v in values]}


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/manifests/train_pilot.jsonl")
    parser.add_argument("--train-metadata", type=Path,
                        default=ROOT / "data/manifests/train_pilot.metadata.json")
    parser.add_argument("--dev-manifest", type=Path, default=ROOT / "data/manifests/dev_pilot.jsonl")
    parser.add_argument("--dev-metadata", type=Path, default=ROOT / "data/manifests/dev_pilot.metadata.json")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"),
                        help="Root of the feature cache written by acoustic_classifier.py "
                             "(the folder that contains features/; default: $OUTPUT_ROOT)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=ROOT / "configs/labels.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for this run (must not exist)")
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads for the heads (default 1, the reproducible setting)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Write predictions only; run the shared evaluator separately")
    args = parser.parse_args(argv)
    if args.output_root is None:
        parser.error("--output-root or OUTPUT_ROOT is required")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def run(args, provenance_fn=expected_provenance):
    started_at, wall_start = base.utc_now(), time.perf_counter()
    config = load_config(args.config)
    labels = ev.load_labels(args.labels)
    num_labels, seeds = config["head"]["num_labels"], config["seeds"]
    if num_labels != len(labels):
        raise ValueError(f"Config has {num_labels} labels, the label file {len(labels)}")

    train_rows, train_split, train_meta = base.load_split(args.train_manifest, args.train_metadata,
                                                          args.labels, labels)
    dev_rows, dev_split, dev_meta = base.load_split(args.dev_manifest, args.dev_metadata, args.labels, labels)
    if train_split != "train" or dev_split != "dev":
        raise ValueError(f"Expected train and dev manifests, got {train_split} and {dev_split}")
    if {r["clip_id"] for r in train_rows} & {r["clip_id"] for r in dev_rows}:
        raise ValueError("Train and dev share clip IDs")

    # Features: the existing cache only; no encoder, no audio.
    provenance = provenance_fn(clf.feature_config(config["encoder"]))
    if provenance["output_dim"] != config["encoder"]["feature_dim"]:
        raise ValueError(f"Cached dim {provenance['output_dim']} differs from the config")
    cache = ae.FeatureCache(args.output_root, provenance)
    if not (cache.root / "config.json").is_file():
        raise ValueError(f"No feature cache at {cache.root}. Fill it first with "
                         "src/experiments/acoustic_classifier.py (see docs/layer_mixture.md).")
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        missing = cache.missing(base.without_labels(rows))
        if missing:
            raise ValueError(f"[{split}] {len(missing)} of {len(rows)} clips are not cached under this "
                             f"provenance (e.g. {missing[0]['clip_id']}); extract them with "
                             "acoustic_classifier.py --audio-root before running this pilot")
    train_ids, dev_ids = [r["clip_id"] for r in train_rows], [r["clip_id"] for r in dev_rows]
    train_x = cache.load(train_ids).features.astype(np.float32)
    dev_x = cache.load(dev_ids).features.astype(np.float32)
    layer_names = provenance["layer_names"]
    for name, matrix in (("train", train_x), ("dev", dev_x)):
        if matrix.shape[1:] != (len(layer_names), provenance["output_dim"]) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} features have shape {matrix.shape} or nonfinite values")
    train_y = np.array([labels.index(r["label"]) for r in train_rows], dtype=np.int64)
    if len(set(train_y.tolist())) != num_labels:
        raise ValueError("Training features do not cover every label")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    base.write_json(args.output_dir / "config.json", config)
    specs, final = arm_specs(config, layer_names)
    timings = {}

    # 1. Folds and training-only selection.
    sel = config["selection"]
    folds, group_of = make_folds(train_rows, sel["cv_folds"], sel["fold_seed"])
    base.write_json(args.output_dir / "folds.json", {
        "cv_folds": sel["cv_folds"], "fold_seed": sel["fold_seed"], "grouping_fields": list(fs.GROUP_FIELDS),
        "groups": len(set(group_of.values())),
        "clips_per_fold": [int((folds == f).sum()) for f in range(sel["cv_folds"])],
        "labels_missing_from_fit_rows": {str(f): sorted(set(labels) - {labels[y] for y in train_y[folds != f]})
                                         for f in range(sel["cv_folds"])},
        "assignments": [{"clip_id": c, "group_id": group_of[c], "fold": int(f)} for c, f in zip(train_ids, folds)]})
    fits = len(specs) * len(seeds)
    base.progress(f"{len(train_ids)} train / {len(dev_ids)} dev clips, {len(layer_names)} layers; "
                  f"{sel['cv_folds']}-fold CV of {len(specs)} arms x {len(seeds)} seeds "
                  f"({fits * sel['cv_folds']} fits up to {sel['candidate_epochs'][-1]} epochs)")
    tick = time.perf_counter()
    cv_rows = cross_validate(train_x, train_y, folds, specs, config, base.progress)
    timings["cross_validation_sec"] = time.perf_counter() - tick
    base.write_jsonl(args.output_dir / "cv_results.jsonl", cv_rows)
    summary, selected = select(cv_rows, specs, sel["candidate_epochs"])
    aliases = {BASELINE: f"layer_{final}", SELECTED: selected}
    for alias, target in aliases.items():
        summary[alias] = {**summary[target], "alias_of": target}
    selection = {"rule": {k: sel[k] for k in ("metric", "tie_breaks", "candidate_epochs", "cv_folds")},
                 "baseline": {"arm": aliases[BASELINE], "layer_index": final, "layer_name": layer_names[final]},
                 "selected_layer": {"arm": selected, "layer_index": int(selected.split("_")[1]),
                                    "layer_name": layer_names[int(selected.split("_")[1])]},
                 "arms": summary}
    base.write_json(args.output_dir / "selection.json", selection)
    base.progress(f"Selected by training CV: {selected} ({selection['selected_layer']['layer_name']}); "
                  f"baseline {aliases[BASELINE]} ({layer_names[final]})")

    # 2. Final fits on the whole training pilot. Dev labels are read from here on.
    dev_y = np.array([labels.index(r["label"]) for r in dev_rows], dtype=np.int64)
    floor = config["preprocessing"]["std_floor"]
    std_train, std_dev = zscore(train_x, [dev_x], floor)
    y_train = torch.from_numpy(train_y)
    batches_per_epoch = math.ceil(len(train_ids) / config["optimization"]["batch_size"])
    headline = [BASELINE, BASELINE_RAW, SELECTED, *config["mixtures"]]
    results, correct, weights, evaluations = {}, {}, {}, {}
    tick = time.perf_counter()
    for name, spec in specs.items():
        x_train, x_dev = arm_input(std_train, train_x, spec), arm_input(std_dev, dev_x, spec)
        epochs = summary[name]["epochs"]
        out = {"train_accuracy": [], "dev_accuracy": [], "dev_macro_f1": []}
        for seed in seeds:
            model = fit(x_train, y_train, config["optimization"], num_labels, seed, [epochs])
            out["train_accuracy"].append(classification_metrics(train_y, logits_of(model, x_train).argmax(1),
                                                                num_labels)["accuracy"])
            logits = logits_of(model, x_dev)
            metrics = classification_metrics(dev_y, logits.argmax(1), num_labels)
            out["dev_accuracy"].append(metrics["accuracy"])
            out["dev_macro_f1"].append(metrics["macro_f1"])
            correct[(name, seed)] = logits.argmax(1) == dev_y
            if isinstance(model, LayerMixture):
                weights.setdefault(name, {})[str(seed)] = dict(zip(
                    [layer_names[i] for i in spec["layers"]], model.layer_weights().detach().tolist()))
            for arm in [a for a in headline if (aliases.get(a) or a) == name]:
                folder = args.output_dir / "arms" / arm / f"seed_{seed}"
                folder.mkdir(parents=True)
                torch.save({"state_dict": model.state_dict(), "labels": labels, "arm": arm,
                            "layers": [layer_names[i] for i in spec["layers"]],
                            "standardized": spec["standardize"], "epochs": epochs, "seed": seed,
                            "config": config}, folder / "head.pt")
                revision = f"{MODEL_NAME}-{arm}:{ev.sha256_file(folder / 'head.pt')[:16]}"
                write_predictions(folder / "predictions.jsonl", dev_rows, logits, labels, revision)
                if not args.skip_evaluation:
                    evaluation = run_evaluator(args, folder / "predictions.jsonl", folder / "evaluation",
                                               f"{MODEL_NAME}-{arm}-seed{seed}")
                    if abs(evaluation["accuracy"] - metrics["accuracy"]) > 1e-12:
                        raise RuntimeError(f"{arm}: shared evaluator disagrees with in-script accuracy")
                    evaluations.setdefault(arm, {})[str(seed)] = evaluation
        results[name] = out
    timings["final_fits_sec"] = time.perf_counter() - tick
    base.write_json(args.output_dir / "mixture_weights.json", weights)

    def report(arm):
        target = aliases.get(arm, arm)
        spec, res = specs[target], results[target]
        row = {"arm": arm, "trained_as": target, "layers": [layer_names[i] for i in spec["layers"]],
               "standardized": spec["standardize"], "epochs": summary[target]["epochs"],
               "updates": summary[target]["epochs"] * batches_per_epoch,
               "cv_accuracy_mean": summary[target]["cv_accuracy_mean"],
               "cv_accuracy_std": summary[target]["cv_accuracy_std"],
               "cv_macro_f1_mean": summary[target]["cv_macro_f1_mean"],
               "train_accuracy": mean_std(res["train_accuracy"]),
               "dev_accuracy": mean_std(res["dev_accuracy"]), "dev_macro_f1": mean_std(res["dev_macro_f1"])}
        if arm != BASELINE:
            row["paired_vs_b0_final_by_seed"] = {str(s): paired(correct[(target, s)], correct[(aliases[BASELINE], s)])
                                                 for s in seeds}
        if arm in evaluations:
            row["shared_evaluator_by_seed"] = evaluations[arm]
        return row

    arms_report = [report(arm) for arm in headline]
    with (args.output_dir / "layer_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer_index", "layer_name", "epochs", "cv_accuracy_mean", "cv_accuracy_std",
                         "cv_macro_f1_mean", "dev_accuracy_mean", "dev_accuracy_std", "dev_macro_f1_mean"])
        for i, layer in enumerate(layer_names):
            s, r = summary[f"layer_{i}"], results[f"layer_{i}"]
            writer.writerow([i, layer, s["epochs"], f"{s['cv_accuracy_mean']:.6f}", f"{s['cv_accuracy_std']:.6f}",
                             f"{s['cv_macro_f1_mean']:.6f}", f"{np.mean(r['dev_accuracy']):.6f}",
                             f"{np.std(r['dev_accuracy']):.6f}", f"{np.mean(r['dev_macro_f1']):.6f}"])
    base.write_json(args.output_dir / "summary.json", {
        "result_role": config["result_role"],
        "note": ("In-source ADI-20 dev pilot (200 clips; one clip = 0.5 points). Selection used "
                 "training CV only; dev numbers decide what to try next and are not out-of-domain."),
        "arms": arms_report})

    timings["wall_time_sec"] = time.perf_counter() - wall_start
    artifacts = ["config.json", "folds.json", "cv_results.jsonl", "selection.json", "mixture_weights.json",
                 "layer_sweep.csv", "summary.json",
                 *sorted(str(p.relative_to(args.output_dir)) for p in (args.output_dir / "arms").rglob("*")
                         if p.is_file())]
    base.write_json(args.output_dir / "run_record.json", {
        "task": config["task"], "result_role": config["result_role"], "model_name": MODEL_NAME,
        "status": "completed", "command": shlex.join([sys.executable, *sys.argv]),
        "started_at_utc": started_at.isoformat(), "finished_at_utc": base.utc_now().isoformat(),
        "code": {**base.git_state(), "script_sha256": ev.sha256_file(Path(__file__)),
                 "config_sha256": ev.sha256_file(args.config)},
        "data": {"dataset_version": train_meta["dataset_version"], "split_version": train_meta["split_version"],
                 "train_manifest_sha256": train_meta["manifest_sha256"],
                 "dev_manifest_sha256": dev_meta["manifest_sha256"], "labels_sha256": train_meta["labels_sha256"]},
        "features": {"cache_root": str(cache.root), "feature_config_sha256": provenance["feature_config_sha256"],
                     "provenance_sha256": base.sha256_text(base.canonical_json(provenance)),
                     "layers": layer_names, "dim": provenance["output_dim"],
                     "train_cache_keys_sha256": clf.cache_keys_sha256(cache, train_ids),
                     "dev_cache_keys_sha256": clf.cache_keys_sha256(cache, dev_ids),
                     "encoder_loaded": False, "audio_read": False},
        "leakage_controls": {
            "selection": "epochs per arm and the selected layer come from grouped CV on training clips only",
            "standardization": "statistics from the training rows of each fit",
            "dev_labels_used_for": "scoring final fits after selection.json is written; never for choices",
            "grouping_limit": "recording IDs are unknown for most pilot clips, so CV folds may share recordings"},
        "counts": {"train_clips": len(train_ids), "dev_clips": len(dev_ids), "arms_trained": len(specs),
                   "seeds": seeds, "cv_fits": len(cv_rows) // len(sel["candidate_epochs"])},
        "resources": {**{k: round(v, 3) for k, v in timings.items()}, "threads": args.threads,
                      "peak_rss_mib": base.peak_rss_mib()},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(),
                        "packages": {name: version(name) for name in ("numpy", "torch")}},
        "artifacts": {name: ev.sha256_file(args.output_dir / name) for name in artifacts}})

    base.progress("")
    base.progress(f"{'arm':<16} {'layers':<22} {'epochs':>6} {'CV acc':>13} {'dev acc':>13} {'dev F1':>7}")
    for row in arms_report:
        layers = row["layers"][0] if len(row["layers"]) == 1 else f"{len(row['layers'])} layers"
        base.progress(f"{row['arm']:<16} {layers:<22} {row['epochs']:>6} "
                      f"{row['cv_accuracy_mean']:>7.1%}±{row['cv_accuracy_std']:<5.1%} "
                      f"{row['dev_accuracy']['mean']:>7.1%}±{row['dev_accuracy']['std']:<5.1%} "
                      f"{row['dev_macro_f1']['mean']:>7.3f}")
    base.progress(f"Wall time {timings['wall_time_sec']:.1f}s. Outputs: {args.output_dir}")
    return 0


def main(argv=None, provenance_fn=expected_provenance):
    args = parse_args(argv)
    try:
        return run(args, provenance_fn)
    except (OSError, ValueError, RuntimeError, KeyError, FloatingPointError) as error:
        print(f"Layer mixture error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
