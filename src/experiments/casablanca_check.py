#!/usr/bin/env python3
"""Casablanca check: score the frozen Idea 1 heads on Casablanca validation (frozen diagnostic).

1. Features. Casablanca clips go through the same frozen AraBEST-RQ checkpoint, FeatureConfig and
   code path as the ADI-20 pilot (mean downmix, sinc resample to 16 kHz, full clip), into the same
   cache. Cached clips are reused, and the encoder loads only if some clips are missing. Rows the
   frozen manifest marks as audio_error are never decoded; they and any extraction failure get an
   explicit failure row and stay in the denominator.
2. Frozen heads. The heads saved by layer_mixture.py (every headline arm and seed) are loaded
   unchanged, and the z-score statistics are recomputed from the same 500 training clips. Before
   Casablanca is scored, every head must reproduce its saved ADI-20 dev logits, or the run stops.
3. Scoring. 20-way predictions, never masked to the 8 present labels; the shared evaluator scores
   them. Diagnostics: drop from the in-source dev score, per-dialect recall, predictions to absent
   labels, region accuracy, duration buckets, JOR by native sample rate, and a paired comparison
   with b0_final.

Casablanca labels are only scored. No layer, seed, head or threshold is chosen here
(docs/casablanca_validation.md); results are cross-dataset evaluation, not a benchmark score.

Exit codes: 0 completed, 2 completed with extraction failures beyond the manifest's audio_error
rows, 1 setup/contract error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import layer_mixture as lm
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "arabest-rq-frozen-layer-mixture-pilot"
HEADLINE = ("b0_final", "b0_final_raw", "selected_layer", "mixture_all", "mixture_middle")
REPRODUCTION_TOLERANCE = 1e-5
# Coarse regions, fixed before any Casablanca scoring; used for a diagnostic only.
REGIONS = {"MAGHREB": "TUN ALG MOR LIB MAU", "NILE": "EGY SUD", "LEVANT": "LEB SYR JOR PAL",
           "GULF": "KSA KUW QAT BAH UAE OMA", "IRAQ": "IRA", "YEMEN": "YEM", "MSA": "MSA"}
REGION_OF = {code: region for region, codes in REGIONS.items() for code in codes.split()}
DURATION_BUCKETS = ((0, 3), (3, 5), (5, 10), (10, math.inf))


# ---------------------------------------------------------------- inputs

def load_pilot(run_dir):
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"Pilot run {run_dir} did not complete (status {record.get('status')})")
    config = lm.load_config(run_dir / "config.json")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    dev_scores = {row["arm"]: row["dev_accuracy"]["mean"] for row in summary["arms"]}
    return record, config, dev_scores


def load_head(path, layer_names, dim, num_labels):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    indices = [layer_names.index(name) for name in saved["layers"]]
    linear = torch.nn.Linear(dim, num_labels)
    model = linear if len(indices) == 1 else lm.LayerMixture(len(indices), linear)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    spec = {"layers": indices, "standardize": bool(saved["standardized"]), "mixture": len(indices) > 1}
    return model, spec, saved


def run_evaluator(manifest, metadata, labels_path, predictions_path, output_dir, model_name):
    command = [sys.executable, str(ROOT / "src/evaluation/evaluate.py"),
               "--manifest", str(manifest), "--metadata", str(metadata),
               "--predictions", str(predictions_path), "--labels", str(labels_path),
               "--split", "dev", "--model-name", model_name, "--output-dir", str(output_dir)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Shared evaluator failed: {completed.stderr.strip()}")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics, shlex.join(command)


# ---------------------------------------------------------------- diagnostics

def diagnostics(rows, predicted, present, labels):
    """predicted: label or None per manifest row (None = failure, counted as wrong)."""
    gold = [r["label"] for r in rows]
    correct = np.array([p == g for p, g in zip(predicted, gold)])
    scored = [p for p in predicted if p is not None]
    absent = Counter(p for p in scored if p not in present)

    def accuracy(mask):
        mask = np.asarray(mask, bool)
        return {"clips": int(mask.sum()), "accuracy": float(correct[mask].mean()) if mask.any() else None}

    # Failed rows may have no decoded duration or rate; they get their own "unknown" bucket.
    durations = np.array([np.nan if r.get("duration_sec") is None else r["duration_sec"] for r in rows])
    rates = np.array(["unknown" if r.get("sample_rate_hz") is None else str(int(r["sample_rate_hz"]))
                      for r in rows])
    jor = np.array([g == "JOR" for g in gold])
    return {
        "accuracy": float(correct.mean()),
        "recall_by_label": {c: accuracy([g == c for g in gold])["accuracy"] for c in present},
        "predictions_to_absent_labels": {"share_of_all_clips": sum(absent.values()) / len(rows),
                                         "counts": dict(absent.most_common())},
        "predicted_label_counts": dict(Counter(scored).most_common()),
        "region_accuracy": float(np.mean([p is not None and REGION_OF[p] == REGION_OF[g]
                                          for p, g in zip(predicted, gold)])),
        "accuracy_by_duration_sec": {**{f"{lo}-{hi}": accuracy((durations >= lo) & (durations < hi))
                                        for lo, hi in DURATION_BUCKETS},
                                     "unknown": accuracy(np.isnan(durations))},
        "jor_accuracy_by_native_sample_rate": {rate: accuracy(jor & (rates == rate))
                                               for rate in sorted(set(rates[jor].tolist()))},
    }, correct


def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": [float(v) for v in values]}


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed layer_mixture.py run folder (with arms/, selection.json, run_record.json)")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"),
                        help="Root of the feature cache (the folder that contains features/; default: $OUTPUT_ROOT)")
    parser.add_argument("--audio-root", type=Path,
                        help="Casablanca audio folder the manifest's audio_path values are relative to; "
                             "only needed for clips that are not cached yet")
    parser.add_argument("--cache-root", type=Path, default=os.environ.get("CACHE_ROOT"),
                        help="Hugging Face download cache (default: $CACHE_ROOT)")
    parser.add_argument("--device", help="Encoder device for extraction (default: cuda if available)")
    parser.add_argument("--extract-batch-size", type=int, default=1,
                        help="Encoder batch size (default 1, the W1-12 setting; larger batches agree within "
                             "the documented 1e-4 tolerance)")
    parser.add_argument("--chunk-size", type=int, default=128, help="Clips decoded and cached at once")
    parser.add_argument("--casablanca-manifest", type=Path, default=ROOT / "data/manifests/casablanca_validation.jsonl")
    parser.add_argument("--casablanca-metadata", type=Path,
                        default=ROOT / "data/manifests/casablanca_validation.metadata.json")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/manifests/train_pilot.jsonl")
    parser.add_argument("--train-metadata", type=Path, default=ROOT / "data/manifests/train_pilot.metadata.json")
    parser.add_argument("--dev-manifest", type=Path, default=ROOT / "data/manifests/dev_pilot.jsonl")
    parser.add_argument("--dev-metadata", type=Path, default=ROOT / "data/manifests/dev_pilot.metadata.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "configs/labels.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run (must not exist)")
    args = parser.parse_args(argv)
    if args.output_root is None:
        parser.error("--output-root or OUTPUT_ROOT is required")
    if args.extract_batch_size < 1 or args.chunk_size < 1:
        parser.error("--extract-batch-size and --chunk-size must be positive")
    return args


def run(args, load_encoder=ae.AcousticEncoder.from_pretrained, provenance_fn=lm.expected_provenance):
    started_at, wall_start = base.utc_now(), time.perf_counter()
    record, pilot_config, pilot_dev = load_pilot(args.run_dir)
    labels = ev.load_labels(args.labels)
    num_labels, seeds = pilot_config["head"]["num_labels"], pilot_config["seeds"]

    # The pilot's own clips, checked against its record, and the frozen Casablanca manifest.
    train_rows, _, train_meta = base.load_split(args.train_manifest, args.train_metadata, args.labels, labels)
    dev_rows, _, dev_meta = base.load_split(args.dev_manifest, args.dev_metadata, args.labels, labels)
    for name, meta in (("train", train_meta), ("dev", dev_meta)):
        if meta["manifest_sha256"] != record["data"][f"{name}_manifest_sha256"]:
            raise ValueError(f"{name} manifest differs from the one the pilot run used")
    cas_rows, cas_split, cas_meta = base.load_split(args.casablanca_manifest, args.casablanca_metadata,
                                                    args.labels, labels)
    if cas_split != "dev" or cas_meta.get("test_split_used"):
        raise ValueError("Only the frozen Casablanca validation manifest may be scored")
    present = [c for c in labels if c in cas_meta["evaluation_label_sets"]["dev"]]
    if {r["clip_id"] for r in cas_rows} & {r["clip_id"] for r in train_rows + dev_rows}:
        raise ValueError("Casablanca clip IDs overlap the pilot's clip IDs")
    unprepared = [r for r in cas_rows if r.get("preparation_status", "ok") != "ok"]
    ok_rows = [r for r in cas_rows if r.get("preparation_status", "ok") == "ok"]

    # Same features as the pilot: provenance must equal the pilot's.
    feature_config = clf.feature_config(pilot_config["encoder"])
    provenance = provenance_fn(feature_config)
    provenance_sha = base.sha256_text(base.canonical_json(provenance))
    if provenance_sha != record["features"]["provenance_sha256"]:
        raise ValueError("Feature provenance differs from the pilot run's")
    cache = ae.FeatureCache(args.output_root, provenance)
    if not (cache.root / "config.json").is_file():
        raise ValueError(f"No feature cache at {cache.root}")
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        if cache.missing(base.without_labels(rows)):
            raise ValueError(f"[{name}] pilot clips are missing from the cache at {cache.root}")
    train_ids, dev_ids = [r["clip_id"] for r in train_rows], [r["clip_id"] for r in dev_rows]
    for name, ids in (("train", train_ids), ("dev", dev_ids)):
        if clf.cache_keys_sha256(cache, ids) != record["features"][f"{name}_cache_keys_sha256"]:
            raise ValueError(f"[{name}] cache entries differ from the ones the pilot run read")

    todo = cache.missing(base.without_labels(ok_rows))
    if todo and (args.audio_root is None or not args.audio_root.is_dir()):
        raise ValueError(f"{len(todo)} Casablanca clips are not cached; pass the Casablanca --audio-root")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    timings, extraction = {}, {"encoder_loaded": False}
    failures = [{"clip_id": r["clip_id"], "status": "audio_error", "error_code": "manifest_audio_error",
                 "message": "preparation_status in the frozen manifest"} for r in unprepared]

    # 1. Extract only what is not cached yet.
    if todo:
        tick = time.perf_counter()
        encoder = load_encoder(feature_config, device=args.device, cache_dir=args.cache_root)
        if base.sha256_text(base.canonical_json(json.loads(json.dumps(encoder.provenance())))) != provenance_sha:
            raise ValueError("Loaded encoder provenance differs from the pilot run's")
        frozen = clf.frozen_summary(encoder)
        state_before = clf.state_sha256(encoder)
        timings["encoder_load_sec"] = time.perf_counter() - tick
        device = encoder.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        extracted, extraction_failures = clf.prepare_features(
            encoder, cache, base.without_labels(ok_rows), args.audio_root,
            {"batch_size": args.extract_batch_size, "chunk_size": args.chunk_size}, "casablanca")
        timings["feature_extraction_sec"] = time.perf_counter() - tick
        frozen.update(state_sha256_before=state_before, state_sha256_after=clf.state_sha256(encoder))
        if frozen["state_sha256_before"] != frozen["state_sha256_after"]:
            raise RuntimeError("Encoder parameters changed during feature extraction")
        duration = {r["clip_id"]: r["duration_sec"] for r in ok_rows}
        extraction = {"encoder_loaded": True, "device": str(device), "frozen": frozen,
                      "batch_size": args.extract_batch_size, "extracted": len(extracted),
                      "extracted_audio_sec": round(sum(duration[c] for c in extracted), 3),
                      "peak_gpu_memory_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                          if device.type == "cuda" else None),
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        failures += extraction_failures
        del encoder
    base.write_jsonl(args.output_dir / "failures.jsonl", failures)
    failed = {f["clip_id"] for f in failures}
    scored_ids = [r["clip_id"] for r in cas_rows if r["clip_id"] not in failed]
    base.progress(f"Casablanca: {len(cas_rows)} clips, {len(scored_ids)} with features, {len(failed)} failures "
                  f"({len(unprepared)} marked audio_error in the manifest); "
                  f"{extraction.get('extracted', 0)} extracted this run")

    # 2. Frozen heads with the pilot's preprocessing, checked against the saved dev logits.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tick = time.perf_counter()
    train_x = cache.load(train_ids).features.astype(np.float32)
    dev_x = cache.load(dev_ids).features.astype(np.float32)
    cas_x = cache.load(scored_ids).features.astype(np.float32)
    std_train, std_dev, std_cas = lm.zscore(train_x, [dev_x, cas_x], pilot_config["preprocessing"]["std_floor"])
    del std_train
    timings["feature_load_sec"] = time.perf_counter() - tick
    layer_names, dim = provenance["layer_names"], provenance["output_dim"]
    position = {cid: i for i, cid in enumerate(scored_ids)}
    failure_of = {f["clip_id"]: f for f in failures}
    reproduction, results, correct_by = {}, {}, {}
    for arm in HEADLINE:
        per_seed = []
        for seed in seeds:
            folder = args.run_dir / "arms" / arm / f"seed_{seed}"
            model, spec, saved = load_head(folder / "head.pt", layer_names, dim, num_labels)
            if saved["arm"] != arm or saved["seed"] != seed:
                raise ValueError(f"{folder}/head.pt belongs to {saved['arm']} seed {saved['seed']}")
            saved_dev = ev.read_jsonl(folder / "predictions.jsonl")
            if [p["clip_id"] for p in saved_dev] != dev_ids:
                raise ValueError(f"{folder}: saved dev predictions are not in dev manifest order")
            dev_logits = lm.logits_of(model, lm.arm_input(std_dev, dev_x, spec))
            gap = float(np.abs(dev_logits - np.array([p["scores"] for p in saved_dev])).max())
            same_labels = [labels[int(i)] for i in dev_logits.argmax(1)] == [p["predicted_label"] for p in saved_dev]
            reproduction[f"{arm}/seed_{seed}"] = {"max_abs_logit_difference": gap, "same_predictions": same_labels}
            if gap > REPRODUCTION_TOLERANCE or not same_labels:
                raise RuntimeError(f"{arm} seed {seed} does not reproduce its saved ADI-20 dev logits "
                                   f"(max difference {gap:.3g}); the heads or preprocessing changed")
            logits = lm.logits_of(model, lm.arm_input(std_cas, cas_x, spec))
            revision = saved_dev[0]["model_revision"]
            prediction_rows, predicted = [], []
            for r in cas_rows:
                cid = r["clip_id"]
                if cid in failure_of:
                    prediction_rows.append({"clip_id": cid, "predicted_label": None, "model_revision": revision,
                                            "status": failure_of[cid]["status"],
                                            "error_code": failure_of[cid]["error_code"]})
                    predicted.append(None)
                else:
                    row = logits[position[cid]]
                    label = labels[int(np.argmax(row))]
                    prediction_rows.append({"clip_id": cid, "predicted_label": label, "model_revision": revision,
                                            "status": "ok", "scores": [float(v) for v in row],
                                            "score_labels": labels, "score_type": "logits"})
                    predicted.append(label)
            out = args.output_dir / "arms" / arm / f"seed_{seed}"
            out.mkdir(parents=True)
            base.write_jsonl(out / "predictions.jsonl", prediction_rows)
            ev.validate_prediction_fields(ev.read_jsonl(out / "predictions.jsonl"), labels)
            metrics, command = run_evaluator(args.casablanca_manifest, args.casablanca_metadata, args.labels,
                                             out / "predictions.jsonl", out / "evaluation",
                                             f"{MODEL_NAME}-{arm}-seed{seed}-casablanca")
            diag, correct = diagnostics(cas_rows, predicted, present, labels)
            if abs(diag["accuracy"] - metrics["accuracy"]) > 1e-12:
                raise RuntimeError(f"{arm}: shared evaluator disagrees with in-script accuracy")
            correct_by[(arm, seed)] = correct
            per_seed.append({"seed": seed, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                             "macro_f1_labels": metrics["macro_f1_labels"], "evaluator_command": command,
                             **{k: v for k, v in diag.items() if k != "accuracy"}})
        results[arm] = per_seed
        base.progress(f"{arm}: scored {len(seeds)} seeds; reproduction max difference "
                      f"{max(reproduction[f'{arm}/seed_{s}']['max_abs_logit_difference'] for s in seeds):.2g}")
    timings["scoring_sec"] = time.perf_counter() - tick
    base.write_json(args.output_dir / "reproduction.json", {"tolerance": REPRODUCTION_TOLERANCE, "heads": reproduction})

    report = []
    for arm in HEADLINE:
        per_seed = results[arm]
        accuracy = mean_std([s["accuracy"] for s in per_seed])
        row = {"arm": arm, "in_source_dev_accuracy": pilot_dev[arm], "casablanca_accuracy": accuracy,
               "drop_points": 100 * (pilot_dev[arm] - accuracy["mean"]),
               "casablanca_macro_f1": mean_std([s["macro_f1"] for s in per_seed]),
               "region_accuracy": mean_std([s["region_accuracy"] for s in per_seed]),
               "absent_label_share": mean_std([s["predictions_to_absent_labels"]["share_of_all_clips"]
                                               for s in per_seed]),
               "recall_by_label_mean": {c: (float(np.mean([s["recall_by_label"][c] for s in per_seed]))
                                            if per_seed[0]["recall_by_label"][c] is not None else None)
                                        for c in present},
               "per_seed": per_seed}
        if arm != "b0_final":
            row["paired_vs_b0_final_by_seed"] = {str(s): lm.paired(correct_by[(arm, s)], correct_by[("b0_final", s)])
                                                 for s in seeds}
        report.append(row)
    base.write_json(args.output_dir / "summary.json", {
        "result_role": "cross_dataset_frozen_diagnostic",
        "note": ("Frozen heads trained on 500 ADI-20 clips, scored on Casablanca validation (TV drama, "
                 "8 present labels, 20-way outputs, failures counted as errors). Not used for any choice; "
                 "not a benchmark score."),
        "present_labels": present, "clips": len(cas_rows), "failures": len(failures), "arms": report})

    timings["wall_time_sec"] = time.perf_counter() - wall_start
    artifacts = ["failures.jsonl", "reproduction.json", "summary.json",
                 *sorted(str(p.relative_to(args.output_dir)) for p in (args.output_dir / "arms").rglob("*")
                         if p.is_file())]
    extra_failures = len(failures) - len(unprepared)
    base.write_json(args.output_dir / "run_record.json", {
        "task": "Casablanca check of the Idea 1 frozen heads", "result_role": "cross_dataset_frozen_diagnostic",
        "status": "completed_with_clip_failures" if extra_failures else "completed",
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at_utc": started_at.isoformat(), "finished_at_utc": base.utc_now().isoformat(),
        "code": {**base.git_state(), "script_sha256": ev.sha256_file(Path(__file__)),
                 "layer_mixture_sha256": ev.sha256_file(Path(lm.__file__)),
                 "encoder_code_sha256": ev.sha256_file(Path(ae.__file__))},
        "pilot_run": {"run_dir": str(args.run_dir), "run_record_sha256": ev.sha256_file(args.run_dir / "run_record.json"),
                      "config_sha256": ev.sha256_file(args.run_dir / "config.json")},
        "data": {"casablanca_manifest_sha256": cas_meta["manifest_sha256"],
                 "casablanca_split_version": cas_meta["split_version"],
                 "casablanca_dataset_version": cas_meta["dataset_version"],
                 "train_manifest_sha256": train_meta["manifest_sha256"],
                 "dev_manifest_sha256": dev_meta["manifest_sha256"], "labels_sha256": train_meta["labels_sha256"],
                 "audio_root": str(args.audio_root) if args.audio_root else None},
        "features": {"cache_root": str(cache.root), "provenance_sha256": provenance_sha,
                     "casablanca_cache_keys_sha256": clf.cache_keys_sha256(cache, scored_ids),
                     "extraction": extraction},
        "rules": {"casablanca_labels_used_for": "scoring only; no layer, seed, head or threshold is chosen here",
                  "outputs": "all 20 labels; predictions to the 12 absent labels count as errors",
                  "failures": "explicit rows, counted as errors in the 6,819-clip denominator",
                  "test_split": "not read"},
        "counts": {"clips": len(cas_rows), "scored": len(scored_ids), "failures": len(failures),
                   "manifest_audio_error_rows": len(unprepared)},
        "resources": {**{k: round(v, 3) for k, v in timings.items()}, "peak_rss_mib": base.peak_rss_mib()},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(),
                        "packages": {name: version(name) for name in ("numpy", "torch")}},
        "artifacts": {name: ev.sha256_file(args.output_dir / name) for name in artifacts}})

    base.progress("")
    base.progress(f"{'arm':<16} {'ADI-20 dev':>10} {'Casablanca':>15} {'drop':>6} {'macro-F1':>8} "
                  f"{'region':>7} {'absent':>7}")
    for row in report:
        acc = row["casablanca_accuracy"]
        base.progress(f"{row['arm']:<16} {row['in_source_dev_accuracy']:>10.1%} "
                      f"{acc['mean']:>8.1%}±{acc['std']:<5.1%} {row['drop_points']:>5.1f} "
                      f"{row['casablanca_macro_f1']['mean']:>8.3f} {row['region_accuracy']['mean']:>7.1%} "
                      f"{row['absent_label_share']['mean']:>7.1%}")
    base.progress("drop = points lost from ADI-20 dev; region = right region; absent = predicted one of the "
                  "12 labels not in Casablanca")
    base.progress(f"Wall time {timings['wall_time_sec']:.1f}s. Outputs: {args.output_dir}")
    return 2 if extra_failures else 0


def main(argv=None, load_encoder=ae.AcousticEncoder.from_pretrained, provenance_fn=lm.expected_provenance):
    args = parse_args(argv)
    try:
        return run(args, load_encoder, provenance_fn)
    except (OSError, ValueError, RuntimeError, KeyError, FloatingPointError) as error:
        print(f"Casablanca check error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
