"""Synthetic, offline checks for Casablanca validation preparation. Nothing is downloaded."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    """Load by file path: an installed top-level `src` or `tests` package (as on Colab) would shadow ours."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


casablanca = load_module("prepare_casablanca", ROOT / "src/data/prepare_casablanca.py")
grouping = load_module("analyze_casablanca_grouping", ROOT / "src/data/analyze_casablanca_grouping.py")
evaluate = load_module("casablanca_test_evaluate", ROOT / "src/evaluation/evaluate.py")
compute_metrics, join_predictions = evaluate.compute_metrics, evaluate.join_predictions
load_evaluation_metadata, load_labels, read_jsonl = (evaluate.load_evaluation_metadata, evaluate.load_labels,
                                                     evaluate.read_jsonl)
validate_prediction_fields, validate_references = evaluate.validate_prediction_fields, evaluate.validate_references

LABELS = json.loads((ROOT / "configs/labels.json").read_text())["label_order"]
SHARDS = {config: (1 if config == "Jordan" else 2) for config in casablanca.CONFIG_TO_LABEL}


def wav_payload(offset=0, stereo=False, identical=True):
    samples = 0.2 * np.sin(np.arange(4410) * 0.05 + offset)
    if stereo:
        samples = np.stack([samples, samples if identical else 0.5 * samples], axis=1)
    output = io.BytesIO()
    sf.write(output, samples, 44100, format="WAV", subtype="PCM_16")
    return output.getvalue()


class CasablancaPreparationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.raw = self.root / "raw"
        self.shared = wav_payload(99, stereo=True)

    def tearDown(self):
        self.folder.cleanup()

    def make_release(self):
        files, counts, offset = [], {}, 0
        for config, total in SHARDS.items():
            for index in range(total):
                name = f"{config}/validation-{index:05d}-of-{total:05d}.parquet"
                rows = []
                for row in range(2):
                    offset += 1
                    payload = wav_payload(offset, stereo=row == 1, identical=offset % 4 == 0)
                    if config == "Algeria" and index == 0 and row == 0:
                        payload = b"broken audio"
                    if (config, index, row) in {("Egypt", 0, 0), ("Yemen", 1, 1)}:
                        payload = self.shared  # cross-label exact duplicate
                    rows.append({"audio": {"bytes": payload, "path": f"{offset}.wav"},
                                 "seg_id": f"{100 + offset // 3}_{row}", "transcription": "نص سري",
                                 "gender": "F", "duration": 0.2 if offset == 5 else 0.1})
                (self.raw / name).parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), self.raw / name)
                files.append({"path": name, "bytes": (self.raw / name).stat().st_size})
                counts[config] = counts.get(config, 0) + len(rows)
        return files, counts

    def adi20_manifest(self, audio_sha256):
        path = self.root / "adi20_train.jsonl"
        path.write_text(json.dumps({"clip_id": "adi-1", "split": "train", "label": "EGY",
                                    "audio_sha256": audio_sha256, "pcm_sha256": "x", "source_id": None}) + "\n")
        return path

    def run_prepare(self, files, counts, adi20):
        args = casablanca.parse_args([
            "--data-root", str(self.root / "data"), "--raw-dir", str(self.raw), "--skip-download",
            "--manifest-dir", str(self.root / "manifests"), "--report-dir", str(self.root / "report"),
            "--adi20-manifests", str(adi20)])
        with patch.object(casablanca, "hub_inventory", return_value=(casablanca.validate_inventory(files), "test")), \
                patch.object(casablanca, "PUBLISHED_VALIDATION_COUNTS", counts), \
                contextlib.redirect_stdout(io.StringIO()):
            return casablanca.run(args)

    def test_inventory_refuses_test_and_incomplete_releases(self):
        files, _ = self.make_release()
        self.assertEqual(len(casablanca.validate_inventory(files)), 15)
        with self.assertRaisesRegex(ValueError, "Test files"):
            casablanca.validate_inventory(files + [{"path": "Egypt/test-00000-of-00002.parquet", "bytes": 1}])
        with self.assertRaisesRegex(ValueError, "Expected 15"):
            casablanca.validate_inventory(files[:-1])
        renamed = [dict(e, path=e["path"].replace("Yemen", "Oman")) for e in files]
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            casablanca.validate_inventory(renamed)

    def test_frozen_manifest_audit_and_determinism(self):
        files, counts = self.make_release()
        adi20 = self.adi20_manifest(__import__("hashlib").sha256(self.shared).hexdigest())
        audit = self.run_prepare(files, counts, adi20)
        manifest = self.root / "manifests/casablanca_validation.jsonl"
        rows = read_jsonl(manifest)
        metadata = json.loads(manifest.with_suffix(".metadata.json").read_text())
        self.assertEqual(len(rows), 30)
        self.assertEqual(metadata["evaluation_label_sets"]["dev"],
                         ["ALG", "EGY", "JOR", "MAU", "MOR", "PAL", "UAE", "YEM"])
        self.assertEqual({r["label"] for r in rows}, set(casablanca.CONFIG_TO_LABEL.values()))
        for row in rows:
            self.assertNotIn("transcription", row)
            self.assertNotIn("gender", row)
            self.assertIsNone(row["source_id"])
            self.assertIsNone(row["speaker_id"])
            self.assertEqual(row["split"], "dev")
        broken = [r for r in rows if r["preparation_status"] == "audio_error"]
        self.assertEqual(len(broken), 1)
        self.assertEqual(broken[0]["label"], "ALG")
        self.assertTrue((self.root / "data" / metadata["audio_root"] / broken[0]["audio_path"]).is_file())
        ok = [r for r in rows if r["preparation_status"] == "ok"]
        self.assertEqual({r["sample_rate_hz"] for r in ok}, {44100})
        self.assertTrue(any(r["channels_identical"] is False for r in ok))
        stored = self.root / "data" / metadata["audio_root"] / ok[0]["audio_path"]
        self.assertEqual(__import__("hashlib").sha256(stored.read_bytes()).hexdigest(), ok[0]["audio_sha256"])
        self.assertEqual(audit["exact_duplicate_label_conflicts"], 2)  # encoded and decoded groups
        self.assertEqual(audit["adi20_overlap_findings"], {"encoded_exact": 2})
        self.assertEqual(audit["duration_mismatches_over_tolerance"], 1)
        self.assertEqual(len(read_jsonl(self.root / "report/exclusions.jsonl")), 1)

        freeze = json.loads((self.root / "manifests/casablanca_validation_freeze.json").read_text())
        before = (manifest.read_bytes(), manifest.with_suffix(".metadata.json").read_bytes(),
                  (self.root / "manifests/casablanca_validation_freeze.json").read_bytes())
        self.run_prepare(files, counts, adi20)
        after = (manifest.read_bytes(), manifest.with_suffix(".metadata.json").read_bytes(),
                 (self.root / "manifests/casablanca_validation_freeze.json").read_bytes())
        self.assertEqual(before, after)
        self.assertEqual(freeze["clip_ids_in_manifest_order"], [r["clip_id"] for r in rows])

    def test_shared_evaluator_accepts_manifest_and_keeps_unsupported_errors(self):
        files, counts = self.make_release()
        self.run_prepare(files, counts, self.adi20_manifest("none"))
        manifest = self.root / "manifests/casablanca_validation.jsonl"
        labels = load_labels(ROOT / "configs/labels.json")
        references = validate_references(read_jsonl(manifest), labels, "dev")
        predictions = []
        for clip_id, row in references.items():
            if row["preparation_status"] == "audio_error":
                predictions.append({"clip_id": clip_id, "predicted_label": None, "model_revision": "t",
                                    "status": "audio_error"})
            else:  # an unsupported label is a valid output and an error
                predictions.append({"clip_id": clip_id, "predicted_label": "LEB" if row["label"] == "JOR"
                                    else row["label"], "model_revision": "t", "status": "ok"})
        metrics = compute_metrics(join_predictions(references, validate_prediction_fields(predictions, labels)),
                                  labels)
        load_evaluation_metadata(manifest.with_suffix(".metadata.json"), manifest, ROOT / "configs/labels.json",
                                 "dev", metrics["macro_f1_labels"])
        self.assertEqual(len(metrics["macro_f1_labels"]), 8)
        self.assertAlmostEqual(metrics["accuracy"], 27 / 30)  # 2 JOR->LEB errors + 1 audio_error

    def test_episode_and_parent_segment_groups_from_original_path(self):
        parsed = grouping.parse_original_path("12 - مسلسل الشريكان الحلقة 12_338.92_343.54_9121_1.wav")
        self.assertEqual(parsed, {"episode_file": "12 - مسلسل الشريكان الحلقة 12", "window_start_sec": 338.92,
                                  "window_end_sec": 343.54, "seg_id": "9121_1"})
        self.assertIsNone(grouping.parse_original_path("no_window.wav"))
        rows = [{"clip_id": str(i), "label": "JOR", "source_config": "Jordan", "seg_id": seg,
                 "original_path": f"ep 1_10.0_12.0_{seg}.wav"} for i, seg in enumerate(["7_1", "7_2", "8_1"])]
        rows.append({"clip_id": "x", "label": "JOR", "source_config": "Jordan", "seg_id": "9_1",
                     "original_path": "ep 1_1.0_2.0_9_2.wav"})  # seg_id disagrees with the file name
        summary, assignments = grouping.analyze(rows)
        self.assertEqual(summary["unparsed_clip_ids"], ["x"])
        self.assertEqual(summary["episode_groups"], 1)
        self.assertEqual(summary["parent_segment_groups"], 2)
        self.assertEqual(assignments[0]["parent_segment_group"], assignments[1]["parent_segment_group"])

    def test_count_mismatch_stops_before_freezing(self):
        files, counts = self.make_release()
        counts = dict(counts, Egypt=counts["Egypt"] + 1)
        with self.assertRaisesRegex(ValueError, "published counts"):
            self.run_prepare(files, counts, self.adi20_manifest("none"))
        self.assertFalse((self.root / "manifests/casablanca_validation.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
