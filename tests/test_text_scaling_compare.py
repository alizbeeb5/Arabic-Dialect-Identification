"""Verify saved scaling evidence and reject corrupted evidence without model inference."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("text_scaling_compare", ROOT / "src/analysis/text_scaling_compare.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
ev = comparison.ev


class TextScalingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "predictions.jsonl"
        self.labels = ev.load_labels(ROOT / "configs/labels.json")
        self.references = ev.validate_references(ev.read_jsonl(comparison.DEV), self.labels, "dev")
        self.evidence = ROOT / "reports/week2_text_scaling"
        self.metrics = json.loads((self.evidence / "dev_500_metrics.json").read_text())
        self.path.write_bytes((self.evidence / "dev_500_predictions.jsonl").read_bytes())

    def check(self, metrics=None):
        return comparison.checked_predictions(self.path, metrics or self.metrics,
                                              self.labels, self.references)

    def test_both_committed_arms_and_week1_probabilities(self):
        predictions = {}
        for arm, correct in (("500", 50), ("2000", 75)):
            path = self.evidence / f"dev_{arm}_predictions.jsonl"
            metrics = json.loads((self.evidence / f"dev_{arm}_metrics.json").read_text())
            predictions[arm] = comparison.checked_predictions(path, metrics, self.labels, self.references)
            self.assertEqual(metrics["correct"], correct)
        week1 = ev.validate_prediction_fields(ev.read_jsonl(
            ROOT / "reports/week1_evidence/text_predictions.jsonl"), self.labels)
        for clip_id in week1:
            self.assertEqual(predictions["500"][clip_id]["scores"], week1[clip_id]["scores"])
        gained = sum(predictions["500"][c]["predicted_label"] != r["label"] and
                     predictions["2000"][c]["predicted_label"] == r["label"]
                     for c, r in self.references.items())
        lost = sum(predictions["500"][c]["predicted_label"] == r["label"] and
                   predictions["2000"][c]["predicted_label"] != r["label"]
                   for c, r in self.references.items())
        self.assertEqual((gained, lost), (32, 7))

    def test_duplicate_rejected_even_with_updated_hash(self):
        first = self.path.read_text().splitlines()[0]
        with self.path.open("a") as stream:
            stream.write(first + "\n")
        metrics = copy.deepcopy(self.metrics)
        metrics["predictions_sha256"] = ev.sha256_file(self.path)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.check(metrics)

    def test_stale_prediction_hash_rejected(self):
        with self.path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.check()

    def test_tampered_f1_rejected_even_when_correct_count_matches(self):
        metrics = copy.deepcopy(self.metrics)
        metrics["macro_f1"] = 0.99
        with self.assertRaisesRegex(ValueError, "macro_f1"):
            self.check(metrics)

    def test_checkpoint_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            comparison.checked_predictions(self.path, self.metrics, self.labels,
                                           self.references, "sha256:another-checkpoint")

    def test_run_loader_accepts_current_evidence_and_rejects_wrong_manifest(self):
        run = self.path.parent
        (run / "metrics.json").write_text(json.dumps(self.metrics))
        (run / "run_record.json").write_text(json.dumps({"model_revision": self.metrics["model_revision"]}))
        _, _, predictions = comparison.load(run, self.labels, comparison.DEV, "dev")
        self.assertEqual(len(predictions), 200)
        stale = {**self.metrics, "manifest_sha256": "another-manifest"}
        (run / "metrics.json").write_text(json.dumps(stale))
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            comparison.load(run, self.labels, comparison.DEV, "dev")

    def test_notebook_installs_the_reviewed_helpers(self):
        notebook = json.loads((ROOT / "notebooks/W2_02_Text_Scaling.ipynb").read_text())
        for name, folder in (("merge_text_exports", "data"), ("text_scaling_compare", "analysis")):
            prefix = f"%%writefile /content/{name}.py\n"
            sources = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                       for c in notebook["cells"]]
            cell = next(s for s in sources if s.startswith(prefix))
            self.assertEqual(cell[len(prefix):], (ROOT / f"src/{folder}/{name}.py").read_text())

    def test_historical_generator_is_explicitly_unverified(self):
        receipt = json.loads((self.evidence / "review_provenance.json").read_text())
        summary = self.evidence / "train2000_export_summary.json"
        self.assertEqual(receipt["historical_summary_sha256"], ev.sha256_file(summary))
        self.assertEqual(receipt["historical_generator_status"], "source_not_recovered")
        self.assertFalse(receipt["full_training_replay_verified"])
        for path, digest in receipt["reviewed_source_sha256"].items():
            self.assertEqual(ev.sha256_file(ROOT / path), digest)


if __name__ == "__main__":
    unittest.main()
