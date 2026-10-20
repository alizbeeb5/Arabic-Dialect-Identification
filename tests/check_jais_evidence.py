"""Verify a downloaded Task 10 evidence bundle without audio, Torch or a GPU."""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate import (
    compute_metrics, join_predictions, load_evaluation_metadata, load_labels,
    read_jsonl, validate_prediction_fields, validate_references,
)


def check_bundle(path):
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(path) as archive:
        root = Path(folder)
        names = ["inputs/train.jsonl", "inputs/train.metadata.json", "configs/labels.json",
                 "run/predictions.jsonl", "run/raw_outputs.jsonl", "run/run_record.json",
                 "evaluation/metrics.json", "checks.json"]
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
        run, saved, checks = read("run/run_record.json"), read("evaluation/metrics.json"), read("checks.json")
        labels = load_labels(root / "configs/labels.json")
        references = validate_references(read_jsonl(root / "inputs/train.jsonl"), labels, "train")
        predictions = validate_prediction_fields(read_jsonl(root / "run/predictions.jsonl"), labels)
        recomputed = compute_metrics(join_predictions(references, predictions), labels)
        load_evaluation_metadata(root / "inputs/train.metadata.json", root / "inputs/train.jsonl",
                                 root / "configs/labels.json", "train", recomputed["macro_f1_labels"])
        for key, value in recomputed.items():
            if saved[key] != value:
                raise ValueError(f"Saved/recomputed metric differs: {key}")
        digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest("run/predictions.jsonl") != saved["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch")
        if digest("evaluation/metrics.json") != run["evaluation"]["metrics_sha256"]:
            raise ValueError("Metrics hash mismatch")
        if list(predictions) != run["clip_ids"]:
            raise ValueError("Run record clip IDs/order mismatch")
        raw = read_jsonl(root / "run/raw_outputs.jsonl")
        if [r["prediction"] for r in raw] != list(predictions.values()):
            raise ValueError("Raw/standardized output mismatch")
        if checks["cache_second_run"]["model_batches"] != 0:
            raise ValueError("Resume check executed model batches")
        if checks["missing_audio_test"]["failed"] != 1:
            raise ValueError("Missing-audio check did not retain its failure")
        return {
            "bundle_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "clips": recomputed["expected"], "accuracy": recomputed["accuracy"],
            "macro_f1": recomputed["macro_f1"], "macro_f1_labels": recomputed["macro_f1_labels"],
            "coverage": recomputed["coverage"], "exported_run_counts": run["counts"],
            "exported_wall_seconds": run["wall_seconds"],
            "exported_peak_gpu_allocated_gib": run["peak_gpu_allocated_gib"],
            "model_revision": run["model_revision"], "packages": run["packages"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_bundle(args.bundle), indent=2))
