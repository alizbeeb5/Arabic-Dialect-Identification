import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.evaluate import (
    compute_metrics,
    join_predictions,
    load_evaluation_metadata,
    load_labels,
    read_jsonl,
    validate_prediction_fields,
    validate_references,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/evaluation"


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = load_labels(ROOT / "configs/labels.json")
        self.references = validate_references(
            read_jsonl(FIXTURES / "references.jsonl"),
            self.labels,
        )
        self.predictions = read_jsonl(FIXTURES / "predictions.jsonl")

    def score(self, records, *, allow_incomplete=False):
        predictions = validate_prediction_fields(records, self.labels)
        joined = join_predictions(
            self.references,
            predictions,
            allow_incomplete=allow_incomplete,
        )
        return compute_metrics(joined, self.labels)

    def test_hand_calculated_scores(self):
        result = self.score(self.predictions)

        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 7 / 12)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["expected"], 4)
        self.assertIsNone(result["c_avg"]["value"])

        matrix = result["confusion_matrix"]["counts"]
        self.assertEqual(matrix[self.labels.index("EGY")][-1], 1)

    def test_shuffled_rows_do_not_change_metrics(self):
        shuffled = read_jsonl(FIXTURES / "predictions_shuffled.jsonl")
        self.assertEqual(
            self.score(self.predictions),
            self.score(shuffled),
        )

    def test_unsupported_classes_are_explicit(self):
        result = self.score(self.predictions)

        self.assertEqual(result["macro_f1_labels"], ["EGY", "LEB"])
        self.assertEqual(result["per_dialect"]["MSA"]["support"], 0)
        self.assertFalse(result["per_dialect"]["MSA"]["supported"])
        self.assertEqual(len(result["per_dialect"]), 20)

    def test_missing_prediction_policy(self):
        partial = self.predictions[:-1]

        with self.assertRaisesRegex(ValueError, "missing clip IDs"):
            self.score(partial)

        result = self.score(partial, allow_incomplete=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["expected"], 4)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["accuracy"], 0.5)

    def test_duplicate_ids_are_rejected(self):
        duplicated = self.predictions + [self.predictions[0]]

        with self.assertRaisesRegex(ValueError, "duplicate clip_id"):
            self.score(duplicated)

    def test_unexpected_ids_are_rejected(self):
        extra = {**self.predictions[0], "clip_id": "unexpected"}

        with self.assertRaisesRegex(ValueError, "unexpected clip IDs"):
            self.score(self.predictions + [extra])

    def test_incorrect_score_order_is_rejected(self):
        scores = [0.0] * 20
        scores[self.labels.index("LEB")] = 1.0
        bad = {
            **self.predictions[0],
            "scores": scores,
            "score_labels": self.labels[::-1],
            "score_type": "logits",
        }

        with self.assertRaisesRegex(ValueError, "canonical order"):
            self.score([bad, *self.predictions[1:]])

    def test_prediction_outside_supported_set_counts_as_wrong(self):
        changed = {
            **self.predictions[0],
            "predicted_label": "MSA",
        }
        result = self.score([changed, *self.predictions[1:]])

        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["expected"], 4)
        self.assertEqual(result["macro_f1_labels"], ["EGY", "LEB"])

    def test_metadata_rejects_changed_manifest(self):
        with TemporaryDirectory() as folder:
            changed_manifest = Path(folder) / "references.jsonl"
            changed_manifest.write_bytes(
                (FIXTURES / "references.jsonl").read_bytes() + b"\n"
            )

            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                load_evaluation_metadata(
                    FIXTURES / "metadata.json",
                    changed_manifest,
                    ROOT / "configs/labels.json",
                    "dev",
                    ["EGY", "LEB"],
                )


if __name__ == "__main__":
    unittest.main()