import json
from pathlib import Path
import math
import hashlib
import argparse
import csv
from datetime import datetime, timezone

def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(
                    line,
                    parse_constant=_reject_constant,
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            records.append(record)

    if not records:
        raise ValueError(f"{path}: no records found")

    return records


def index_by_clip_id(
    records: list[dict],
    source_name: str,
) -> dict[str, dict]:
    indexed = {}

    for record_number, record in enumerate(records, start=1):
        clip_id = record.get("clip_id")

        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(
                f"{source_name}: record {record_number}: "
                "clip_id must be a nonempty string"
            )

        if clip_id in indexed:
            raise ValueError(
                f"{source_name}: record {record_number}: "
                f"duplicate clip_id {clip_id!r}"
            )

        indexed[clip_id] = record

    return indexed

def load_labels(path: str | Path) -> list[str]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file, parse_constant=_reject_constant)

    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")

    for field, expected in (("num_labels", 20), ("index_base", 0)):
        value = config.get(field)
        if type(value) is not int or value != expected:
            raise ValueError(f"{path}: {field} must be {expected}")

    labels = config.get("label_order")

    if not isinstance(labels, list) or len(labels) != 20:
        raise ValueError(f"{path}: label_order must contain 20 labels")

    if any(
        not isinstance(label, str) or not label.strip()
        for label in labels
    ):
        raise ValueError(f"{path}: labels must be nonempty strings")

    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: label_order contains duplicates")

    expected_mapping = {
        label: index for index, label in enumerate(labels)
    }
    mapping = config.get("label_to_id")

    if (
        not isinstance(mapping, dict)
        or mapping != expected_mapping
        or any(type(value) is not int for value in mapping.values())
    ):
        raise ValueError(f"{path}: label_to_id contradicts label_order")

    expected_reverse = {
        str(index): label for index, label in enumerate(labels)
    }
    if config.get("id_to_label") != expected_reverse:
        raise ValueError(f"{path}: id_to_label contradicts label_order")

    return labels

def validate_prediction_fields(
    records: list[dict],
    labels: list[str],
) -> dict[str, dict]:
    indexed = index_by_clip_id(records, "predictions")
    required = {
        "clip_id", "predicted_label", "model_revision", "status"
    }
    allowed_statuses = {"ok", "audio_error", "inference_error"}
    score_fields = {"scores", "score_labels", "score_type"}
    revisions = set()

    for clip_id, record in indexed.items():
        context = f"prediction {clip_id!r}"

        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"{context}: missing fields {sorted(missing)}"
            )

        revision = record["model_revision"]
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                f"{context}: model_revision must be a nonempty string"
            )
        revisions.add(revision)

        status = record["status"]
        if not isinstance(status, str) or status not in allowed_statuses:
            raise ValueError(f"{context}: invalid status {status!r}")

        predicted_label = record["predicted_label"]

        if status == "ok":
            if (
                not isinstance(predicted_label, str)
                or predicted_label not in labels
            ):
                raise ValueError(
                    f"{context}: status=ok requires a valid dialect label"
                )
        else:
            if predicted_label is not None:
                raise ValueError(
                    f"{context}: failure requires predicted_label=null"
                )

            if score_fields.intersection(record):
                raise ValueError(
                    f"{context}: failure must omit all score fields"
                )
        validate_scores(record, labels)
        if "error_code" in record:
            error_code = record["error_code"]
            if (
                status == "ok"
                or not isinstance(error_code, str)
                or not error_code.strip()
            ):
                raise ValueError(
                    f"{context}: error_code must be a nonempty string "
                    "on a failure record"
                )

    if len(revisions) != 1:
        raise ValueError(
            "predictions: expected exactly one model_revision per file"
        )

    return indexed

def validate_scores(record: dict, labels: list[str]) -> None:
    fields = {"scores", "score_labels", "score_type"}
    present = fields.intersection(record)
    context = f"prediction {record['clip_id']!r}"

    if not present:
        return  # Hard-label-only predictions are allowed.

    if present != fields:
        raise ValueError(
            f"{context}: supply all three score fields together"
        )

    if record["score_labels"] != labels:
        raise ValueError(
            f"{context}: score_labels must match canonical order"
        )

    scores = record["scores"]
    if not isinstance(scores, list) or len(scores) != len(labels):
        raise ValueError(f"{context}: expected {len(labels)} scores")

    if any(type(value) not in (int, float) for value in scores):
        raise ValueError(
            f"{context}: scores must be numbers, not booleans"
        )

    try:
        values = [float(value) for value in scores]
    except OverflowError as error:
        raise ValueError(
            f"{context}: score outside supported numeric range"
        ) from error

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context}: scores must be finite")

    score_type = record["score_type"]
    if not isinstance(score_type, str) or score_type not in {
        "logits", "probabilities", "log_probabilities"
    }:
        raise ValueError(f"{context}: invalid score_type")

    if score_type == "probabilities":
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(
                f"{context}: probabilities must be in [0, 1]"
            )
        if abs(math.fsum(values) - 1.0) > 1e-5:
            raise ValueError(
                f"{context}: probabilities must sum to 1"
            )

    elif score_type == "log_probabilities":
        if any(value > 1e-6 for value in values):
            raise ValueError(
                f"{context}: log probabilities exceed tolerance"
            )

        peak = max(values)
        log_total = peak + math.log(
            math.fsum(math.exp(value - peak) for value in values)
        )
        if abs(log_total) > 1e-5:
            raise ValueError(
                f"{context}: log probabilities are not normalized"
            )

    winning_index = max(
        range(len(scores)),
        key=lambda index: scores[index],
    )
    if record["predicted_label"] != labels[winning_index]:
        raise ValueError(
            f"{context}: predicted_label disagrees with score argmax"
        )


def validate_references(
    records: list[dict],
    labels: list[str],
    split: str = "dev",
) -> dict[str, dict]:
    """Validate evaluation fields and select one labeled split."""
    allowed_splits = {"train", "dev", "test"}
    if split not in allowed_splits:
        raise ValueError(f"Invalid requested split: {split!r}")

    indexed = index_by_clip_id(records, "references")
    selected = {}

    for clip_id, record in indexed.items():
        context = f"reference {clip_id!r}"

        if "label" not in record or "split" not in record:
            raise ValueError(f"{context}: label and split are required")

        record_split = record["split"]
        if (
            not isinstance(record_split, str)
            or record_split not in allowed_splits
        ):
            raise ValueError(f"{context}: invalid split")

        label = record["label"]
        if label is not None and (
            not isinstance(label, str) or label not in labels
        ):
            raise ValueError(f"{context}: invalid reference label")

        if record_split == split:
            if label is None:
                raise ValueError(
                    f"{context}: cannot score an unlabeled reference"
                )
            selected[clip_id] = record

    if not selected:
        raise ValueError(f"references: no labeled clips for split {split!r}")

    return selected


def join_predictions(
    references: dict[str, dict],
    predictions: dict[str, dict],
    *,
    allow_incomplete: bool = False,
) -> list[tuple[dict, dict | None]]:
    unexpected = sorted(predictions.keys() - references.keys())
    if unexpected:
        raise ValueError(
            f"predictions: {len(unexpected)} unexpected clip IDs; "
            f"examples: {unexpected[:5]}"
        )

    missing = sorted(references.keys() - predictions.keys())
    if missing and not allow_incomplete:
        raise ValueError(
            f"predictions: {len(missing)} missing clip IDs; "
            f"examples: {missing[:5]}. "
            "Include explicit failure records for failed clips."
        )

    return [
        (reference, predictions.get(clip_id))
        for clip_id, reference in references.items()
    ]


def compute_metrics(
    joined: list[tuple[dict, dict | None]],
    labels: list[str],
) -> dict:
    if not joined:
        raise ValueError("Cannot score an empty evaluation set")

    label_to_index = {
        label: index for index, label in enumerate(labels)
    }
    size = len(labels)
    matrix = [[0] * (size + 1) for _ in labels]
    successful = 0
    missing = 0
    failures = {"audio_error": 0, "inference_error": 0}

    for reference, prediction in joined:
        row = label_to_index[reference["label"]]

        if prediction is None:
            missing += 1
            column = size
        elif prediction["status"] != "ok":
            failures[prediction["status"]] += 1
            column = size
        else:
            successful += 1
            column = label_to_index[prediction["predicted_label"]]

        matrix[row][column] += 1

    expected = len(joined)
    correct = sum(matrix[index][index] for index in range(size))
    per_dialect = {}
    supported_labels = []

    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)

        precision = (
            true_positive / predicted_count if predicted_count else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * true_positive / (support + predicted_count)
            if support + predicted_count else 0.0
        )

        per_dialect[label] = {
            "support": support,
            "supported": support > 0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if support:
            supported_labels.append(label)

    macro_f1 = math.fsum(
        per_dialect[label]["f1"] for label in supported_labels
    ) / len(supported_labels)

    return {
        "complete": missing == 0,
        "expected": expected,
        "successful": successful,
        "failed": sum(failures.values()),
        "failures_by_status": failures,
        "missing": missing,
        "correct": correct,
        "coverage": successful / expected,
        "accuracy": correct / expected,
        "macro_f1": macro_f1,
        "macro_f1_labels": supported_labels,
        "per_dialect": per_dialect,
        "confusion_matrix": {
            "row_labels": list(labels),
            "column_labels": [*labels, "__failure_or_missing__"],
            "counts": matrix,
        },
        "c_avg": {
            "value": None,
            "reason": "Official C_avg scoring is not implemented or verified.",
        },
    }

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_evaluation_metadata(
    path: str | Path,
    manifest_path: str | Path,
    labels_path: str | Path,
    split: str,
    supported_labels: list[str],
) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file, parse_constant=_reject_constant)

    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: expected a JSON object")

    for field in ("schema_version", "dataset_version", "split_version"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{path}: {field} must be a nonempty string"
            )

    for field, input_path in (
        ("manifest_sha256", manifest_path),
        ("labels_sha256", labels_path),
    ):
        if metadata.get(field) != sha256_file(input_path):
            raise ValueError(
                f"{path}: {field} does not match {input_path}"
            )

    evaluation_sets = metadata.get("evaluation_label_sets")
    if not isinstance(evaluation_sets, dict):
        raise ValueError(f"{path}: evaluation_label_sets is required")

    declared = evaluation_sets.get(split)
    if (
        not isinstance(declared, list)
        or any(not isinstance(label, str) for label in declared)
    ):
        raise ValueError(
            f"{path}: expected a label list for split {split!r}"
        )

    if (
        len(declared) != len(set(declared))
        or set(declared) != set(supported_labels)
    ):
        raise ValueError(
            f"{path}: declared evaluation labels do not match "
            "the labels with positive reference support"
        )

    return metadata

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and score Arabic dialect predictions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, default=Path("configs/labels.json")
    )
    parser.add_argument(
        "--split", choices=["train", "dev", "test"], default="dev"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic only: retain missing predictions as incorrect.",
    )
    args = parser.parse_args()

    if not args.model_name.strip():
        parser.error("--model-name must be nonempty")

    try:
        labels = load_labels(args.labels)
        references = validate_references(
            read_jsonl(args.manifest), labels, args.split
        )
        predictions = validate_prediction_fields(
            read_jsonl(args.predictions), labels
        )
        joined = join_predictions(
            references,
            predictions,
            allow_incomplete=args.allow_incomplete,
        )
        metrics = compute_metrics(joined, labels)

        metadata = load_evaluation_metadata(
            args.metadata,
            args.manifest,
            args.labels,
            args.split,
            metrics["macro_f1_labels"],
        )

        metrics.update({
            "metrics_schema_version": "1.0",
            "dataset_version": metadata["dataset_version"],
            "split_version": metadata["split_version"],
            "split": args.split,
            "model_name": args.model_name,
            "model_revision": next(
                iter(predictions.values())
            )["model_revision"],
            "evaluation_mode": (
                "complete"
                if metrics["complete"]
                else "incomplete_diagnostic"
            ),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": metadata["manifest_sha256"],
            "labels_sha256": metadata["labels_sha256"],
            "predictions_sha256": sha256_file(args.predictions),
            "metadata_sha256": sha256_file(args.metadata),
            "evaluator_revision": sha256_file(Path(__file__)),
        })

        # Use a new directory so earlier evaluation records are preserved.
        args.output_dir.mkdir(parents=True, exist_ok=False)

        metrics_path = args.output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        matrix_path = args.output_dir / "confusion_matrix.csv"
        matrix = metrics["confusion_matrix"]

        with matrix_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["true_label", *matrix["column_labels"]])
            for label, counts in zip(
                matrix["row_labels"], matrix["counts"]
            ):
                writer.writerow([label, *counts])

    except (OSError, ValueError) as error:
        parser.exit(2, f"Evaluation error: {error}\n")

    print(f"Mode: {metrics['evaluation_mode']}")
    print(f"Clips: {metrics['expected']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Coverage: {metrics['coverage']:.2%}")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()