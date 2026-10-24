"""Task 13 smoke selection and audited transcript-only export (standard library).

Inference stays in src.models.jais_adapter; this module never loads a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate as ev
from src.models import jais_adapter as jais

ROOT = Path(__file__).resolve().parents[2]
TAG_PATTERN = re.compile(r"\[(?:" + "|".join(re.escape(k) for k in jais.JAIS_TO_ADI20) + r")\]", re.I)


def write_jsonl(path, rows):
    path.write_bytes(b"".join(jais.json_bytes(row) + b"\n" for row in rows))


def checked_manifest(manifest, metadata, labels_path, split):
    labels = ev.load_labels(labels_path)
    rows = ev.read_jsonl(manifest)
    references = ev.validate_references(rows, labels, split)
    if len(references) != len(rows):
        raise ValueError("Input must contain exactly one requested split")
    supported = [c for c in labels if any(r["label"] == c for r in rows)]
    meta = ev.load_evaluation_metadata(metadata, manifest, labels_path, split, supported)
    return rows, references, labels, meta


def prepare_smoke(manifest, metadata, labels_path, output_dir):
    rows, _, labels, meta = checked_manifest(manifest, metadata, labels_path, "train")
    # Canonical label order; immutable ID sort; no model results influence selection.
    selected = []
    for label in labels:
        choices = sorted((r for r in rows if r["label"] == label), key=lambda r: r["clip_id"])
        if not choices:
            raise ValueError(f"Training pilot has no clips for {label}")
        selected.append(choices[0])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "smoke.jsonl"
    write_jsonl(path, selected)
    smoke_meta = {**meta, "manifest_sha256": jais.sha256_file(path), "clips": len(selected),
                  "label_counts": dict(Counter(r["label"] for r in selected)),
                  "derived_from": {"manifest_sha256": meta["manifest_sha256"],
                                   "split_version": meta["split_version"]},
                  "selection": "first_clip_id_per_canonical_label_from_frozen_train_pilot",
                  "development_only": True}
    jais.atomic_json(output / "smoke.metadata.json", smoke_meta)
    return {"clips": len(selected), "manifest": str(path)}


def transcript_row(record):
    """No gold labels, dialect predictions or source metadata enter this record."""
    prediction = record["prediction"]
    text = record["transcript"]
    code = prediction.get("error_code")
    if prediction["status"] == "audio_error":
        status = "audio_error"
    elif record["refused"] is True:
        status = "refused"
    elif prediction["status"] != "ok" and code != "unrecognized_dialect":
        status = "inference_error"
    elif not isinstance(text, str) or not text.strip():
        status = "empty"
    elif TAG_PATTERN.search(text) or "<|" in text:
        status = "tag_contamination"
    elif record["truncated"] is True:
        status = "truncated"
    elif record["truncated"] is None:
        status = "truncation_unknown"
    else:
        status = "ok"
    # Retain partial text with explicit status; raw evidence remains in original run.
    retained = text if status in {"ok", "truncated", "truncation_unknown"} else None
    return {"clip_id": prediction["clip_id"], "transcript": retained,
            "transcript_status": status, "usable_for_text_model": status == "ok",
            "refused": record["refused"], "truncated": record["truncated"]}


def export_run(manifest, metadata, labels_path, split, run_dir, output_dir):
    rows, references, labels, meta = checked_manifest(manifest, metadata, labels_path, split)
    run_dir = Path(run_dir)
    run = jais.read_json(run_dir / "run_record.json")
    if run.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("Only completed adapter runs can be exported")
    for key, expected in {"manifest_sha256": jais.sha256_file(manifest),
                          "metadata_sha256": jais.sha256_file(metadata),
                          "model_revision": jais.MODEL_REVISION,
                          "model_id": jais.MODEL_ID,
                          "expected": len(rows),
                          "dataset_version": meta["dataset_version"],
                          "split_version": meta["split_version"],
                          "labels_sha256": jais.sha256_file(labels_path)}.items():
        if run.get(key) != expected:
            raise ValueError(f"Run provenance mismatch: {key}")
    for name, key in [("raw_outputs.jsonl", "raw_outputs_sha256"),
                      ("predictions.jsonl", "predictions_sha256")]:
        if jais.sha256_file(run_dir / name) != run.get(key):
            raise ValueError(f"Run output hash mismatch: {name}")
    raw = ev.read_jsonl(run_dir / "raw_outputs.jsonl")
    indexed = {}
    for record in raw:
        clip_id = record.get("prediction", {}).get("clip_id")
        jais.validate_record(record, clip_id, jais.MODEL_REVISION)
        if clip_id in indexed:
            raise ValueError("Duplicate raw clip ID")
        indexed[clip_id] = record
    if set(indexed) != set(references):
        raise ValueError("Raw outputs must exactly match expected clip IDs")
    predictions = ev.validate_prediction_fields(ev.read_jsonl(run_dir / "predictions.jsonl"), labels)
    joined = ev.join_predictions(references, predictions, allow_incomplete=False)
    for clip_id, record in indexed.items():
        if record["prediction"] != predictions[clip_id]:
            raise ValueError("Raw and standardized predictions disagree")
    ordered = [indexed[row["clip_id"]] for row in rows]
    texts = [transcript_row(record) for record in ordered]
    metrics = ev.compute_metrics(joined, labels)
    counts = run.get("counts", {})
    for key, expected in {"expected": len(rows),
                          "successful": sum(p["status"] == "ok" for p in predictions.values()),
                          "failed": sum(p["status"] != "ok" for p in predictions.values())}.items():
        if type(counts.get(key)) is not int or counts[key] != expected:
            raise ValueError(f"Run count mismatch: {key}")
    summary = {
        "schema_version": "1.0", "split": split, "expected": len(rows),
        "manifest_sha256": meta["manifest_sha256"],
        "metadata_sha256": jais.sha256_file(metadata),
        "labels_sha256": jais.sha256_file(labels_path),
        "split_version": meta["split_version"], "dataset_version": meta["dataset_version"],
        "adapter_run_record_sha256": jais.sha256_file(run_dir / "run_record.json"),
        "model_revision": jais.MODEL_REVISION, "exporter_sha256": jais.sha256_file(__file__),
        "transcript_counts": dict(Counter(t["transcript_status"] for t in texts)),
        "dialect_failure_counts": dict(Counter(r["prediction"].get("error_code", "unspecified")
                                               for r in ordered if r["prediction"]["status"] != "ok")),
        "refused": sum(r["refused"] is True for r in ordered),
        "truncated": sum(r["truncated"] is True for r in ordered),
        "truncation_unknown": sum(r["truncated"] is None for r in ordered),
        "resource_record": {k: run.get(k) for k in ["counts", "wall_seconds", "timing_scope",
                                                  "peak_gpu_allocated_gib", "inference_config"]},
        "model_generation_gpu_seconds": None,
        "timing_note": "Adapter wall time includes loading and I/O; GPU-only generation time is not measured. Cache-only runs are not inference-throughput evidence.",
        "text_policy": "One row per expected ID. Default usable text requires nonempty, nonrefused, untruncated parsed text with no generated tag markers. Unknown dialect tags do not invalidate usable text. Task 14 must explicitly handle unusable rows and keep the full dev denominator.",
    }
    seconds = run.get("wall_seconds")
    hits = counts.get("cache_hits")
    summary["cold_end_to_end_clips_per_second"] = (
        len(rows) / seconds if hits == 0 and counts["failed"] == 0
        and isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds > 0 else None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "transcripts.jsonl", texts)
    write_jsonl(output / "targets.jsonl", [{"clip_id": r["clip_id"], "label": r["label"]} for r in rows])
    write_jsonl(output / "diagnostics.jsonl", [
        {"clip_id": r["prediction"]["clip_id"], "dialect_status": r["prediction"]["status"],
         "dialect_error_code": r["prediction"].get("error_code"),
         "native_dialect": r["native_dialect"], "predicted_label": r["prediction"]["predicted_label"],
         "transcript_status": t["transcript_status"], "refused": r["refused"], "truncated": r["truncated"]}
        for r, t in zip(ordered, texts, strict=True)])
    for name in ["transcripts.jsonl", "targets.jsonl", "diagnostics.jsonl"]:
        summary[name + "_sha256"] = jais.sha256_file(output / name)
    jais.atomic_json(output / "summary.json", summary)
    jais.atomic_json(output / "direct_dialect_metrics.json", {
        **metrics, "model_name": "jais-direct-pilot", "model_revision": jais.MODEL_REVISION,
        "manifest_sha256": meta["manifest_sha256"], "predictions_sha256": run["predictions_sha256"],
        "evaluator_sha256": jais.sha256_file(ev.__file__)})
    return {"expected": len(rows), "transcript_counts": summary["transcript_counts"],
            "accuracy": metrics["accuracy"], "output_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare-smoke", "export"]:
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--metadata", type=Path, required=True)
        sub.add_argument("--labels", type=Path, default=ROOT / "configs/labels.json")
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "export":
            sub.add_argument("--run-dir", type=Path, required=True)
            sub.add_argument("--split", choices=["train", "dev"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-smoke":
            result = prepare_smoke(args.manifest, args.metadata, args.labels, args.output_dir)
        else:
            result = export_run(args.manifest, args.metadata, args.labels, args.split,
                                args.run_dir, args.output_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Jais pilot error: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
