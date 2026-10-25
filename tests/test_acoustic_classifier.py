"""Offline W1-12 checks with a tiny random encoder on synthetic audio; no real results are claimed.

Run in the acoustic environment, from the repository root:
    python -m unittest discover -s tests -p 'test_acoustic_classifier.py' -v
"""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from unittest.mock import patch

from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae
from test_mfcc_svm_baseline import write_split

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "configs/labels.json"
LABELS = ev.load_labels(LABELS_PATH)
TINY = ae.Architecture(d_model=32, nhead=4, num_layers=2, d_ffn=64)


def tiny_encoder(feature_config, device=None, cache_dir=None):
    torch.manual_seed(0)
    return ae.AcousticEncoder(feature_config, TINY, device="cpu")


def tiny_config(**overrides):
    config = clf.load_config(clf.DEFAULT_CONFIG)
    config["encoder"]["feature_dim"] = TINY.d_model
    config["optimization"].update(epochs=5, learning_rate=1e-2)
    # One clip per label: 20 points in 32 dimensions are linearly separable.
    config["overfit_check"].update(clips_per_label=1, epochs=300, learning_rate=5e-2)
    for section, values in overrides.items():
        config[section].update(values)
    return config


class HeadTests(unittest.TestCase):
    SETTINGS = {**clf.load_config(clf.DEFAULT_CONFIG)["optimization"], "num_labels": 20}

    def test_zero_logits_give_chance_loss(self):
        head = torch.nn.Linear(4, 20)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        x, y = torch.randn(8, 4), torch.arange(8)
        self.assertAlmostEqual(clf.score_split(head, x, y)["loss"], math.log(20), places=6)

    def test_ties_choose_lowest_canonical_index(self):
        head = torch.nn.Linear(3, 20)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        _, predicted = clf.predict(head, np.ones((2, 3), dtype=np.float32))
        self.assertEqual(predicted.tolist(), [0, 0])

    def test_head_matches_verifier_spec_parameter_count(self):
        head = clf.build_head(848, 20, seed=1)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 16980)
        self.assertTrue(torch.equal(head.bias, torch.zeros(20)))

    def test_training_is_reproducible_and_learns_separable_data(self):
        rng = np.random.default_rng(0)
        centres = rng.standard_normal((20, 16)).astype(np.float32) * 3
        y = np.repeat(np.arange(20), 5)
        x = (centres[y] + 0.1 * rng.standard_normal((100, 16))).astype(np.float32)
        settings = {**self.SETTINGS, "epochs": 40, "learning_rate": 1e-2}
        first, curve = clf.train_head(x, y, settings, seed=3)
        second, _ = clf.train_head(x, y, settings, seed=3)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertLess(curve[-1]["train"]["loss"], curve[0]["train"]["loss"])
        self.assertEqual(curve[-1]["train"]["accuracy"], 1.0)
        self.assertEqual(curve[-1]["updates"], 40)  # 100 clips, batch 256: one partial batch per epoch

    def test_monitor_split_does_not_change_training(self):
        rng = np.random.default_rng(1)
        x, y = rng.standard_normal((30, 8)).astype(np.float32), np.arange(30) % 20
        settings = {**self.SETTINGS, "epochs": 3}
        alone, _ = clf.train_head(x, y, settings, seed=5)
        monitored, curve = clf.train_head(x, y, settings, seed=5, monitor={"dev": (x[:4], y[:4])})
        self.assertTrue(torch.equal(alone.weight, monitored.weight))
        self.assertIn("dev", curve[-1])

    def test_feature_stats_flag_collapsed_features(self):
        stats = clf.feature_stats(np.ones((5, 4), dtype=np.float32))
        self.assertEqual(stats["duplicate_rows"], 4)
        self.assertEqual(stats["constant_dims"], 4)
        self.assertAlmostEqual(stats["mean_offdiag_cosine"], 1.0, places=6)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.train_meta, self.train_rows = write_split(self.root, "train", per_label=2)
        self.dev, self.dev_meta, self.dev_rows = write_split(self.root, "dev", per_label=1)
        self.config_path = self.write_config(tiny_config())

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, config, name="config.json"):
        path = self.root / name
        base.write_json(path, config)
        return path

    def run_classifier(self, name, extra=(), dev=None, dev_meta=None, config=None, audio=True):
        argv = ["--train-manifest", str(self.train), "--train-metadata", str(self.train_meta),
                "--dev-manifest", str(dev or self.dev), "--dev-metadata", str(dev_meta or self.dev_meta),
                "--output-root", str(self.root / "cache"), "--labels", str(LABELS_PATH),
                "--config", str(config or self.config_path), "--output-dir", str(self.root / name), *extra]
        if audio:
            argv += ["--audio-root", str(self.root / "audio")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return clf.main(argv, load_encoder=tiny_encoder)

    def record(self, name):
        return json.loads((self.root / name / "run_record.json").read_text())

    def test_end_to_end_outputs_scores_and_frozen_encoder(self):
        self.assertEqual(self.run_classifier("run"), 0)
        out = self.root / "run"
        for name in ("checkpoint.pt", "predictions.jsonl", "learning_curve.jsonl", "overfit_check.json",
                     "failures.jsonl", "config.json", "run_record.json", "evaluation/metrics.json",
                     "evaluation/confusion_matrix.csv"):
            self.assertTrue((out / name).is_file(), name)
        predictions = ev.read_jsonl(out / "predictions.jsonl")
        self.assertEqual([p["clip_id"] for p in predictions], [r["clip_id"] for r in self.dev_rows])
        for row in predictions:
            self.assertEqual(row["score_labels"], LABELS)
            self.assertEqual(row["score_type"], "logits")
            self.assertEqual(row["predicted_label"], LABELS[int(np.argmax(row["scores"]))])
        record = self.record("run")
        frozen = record["encoder"]["frozen"]
        self.assertEqual(frozen["trainable_parameters"], 0)
        self.assertFalse(frozen["training_mode"])
        self.assertEqual(frozen["state_sha256_before"], frozen["state_sha256_after"])
        self.assertEqual(record["encoder"]["revision"], ae.REVISION)
        self.assertEqual(record["features"]["layer_name"], "final")
        self.assertEqual(record["features"]["extracted_this_run"], {"train": 40, "dev": 20})
        self.assertTrue(record["overfit_check"]["passed"])
        self.assertEqual(record["development_evaluation"]["evaluation_mode"], "complete")
        self.assertEqual(record["development_evaluation"]["expected"], 20)
        curve = ev.read_jsonl(out / "learning_curve.jsonl")
        self.assertEqual([row["epoch"] for row in curve], list(range(6)))
        # The saved checkpoint reproduces the written logits from the cached features.
        saved = torch.load(out / "checkpoint.pt", weights_only=False)
        head = torch.nn.Linear(saved["feature_dim"], 20)
        head.load_state_dict(saved["state_dict"])
        cache = ae.FeatureCache(self.root / "cache", tiny_encoder(clf.feature_config(saved["config"]["encoder"]))
                                .provenance())
        position = cache.provenance["layer_names"].index("final")
        logits, _ = clf.predict(head, clf.load_layer(cache, [p["clip_id"] for p in predictions], position))
        np.testing.assert_allclose(logits, [p["scores"] for p in predictions], rtol=0, atol=1e-6)

    def test_rerun_reuses_cache_without_audio_and_regenerates_identical_predictions(self):
        self.assertEqual(self.run_classifier("first"), 0)
        self.assertEqual(self.run_classifier("second", audio=False), 0)
        first, second = self.record("first"), self.record("second")
        self.assertEqual(second["features"]["extracted_this_run"], {"train": 0, "dev": 0})
        for key in ("train_cache_keys_sha256", "dev_cache_keys_sha256", "provenance_sha256"):
            self.assertEqual(first["features"][key], second["features"][key])
        self.assertEqual((self.root / "first/predictions.jsonl").read_bytes(),
                         (self.root / "second/predictions.jsonl").read_bytes())

    def test_uncached_clips_without_audio_root_are_a_setup_error(self):
        self.assertEqual(self.run_classifier("run", audio=False), 1)

    def test_dev_labels_do_not_influence_checkpoint_or_predictions(self):
        self.assertEqual(self.run_classifier("original", ["--skip-evaluation"]), 0)
        shuffled = [dict(r) for r in self.dev_rows]
        permuted = list(np.random.default_rng(7).permutation([r["label"] for r in self.dev_rows]))
        for row, label in zip(shuffled, permuted):
            row["label"] = str(label)
        dev2 = self.root / "dev_shuffled.jsonl"
        base.write_jsonl(dev2, shuffled)
        meta2 = self.root / "dev_shuffled.metadata.json"
        meta = json.loads(self.dev_meta.read_text())
        meta["manifest_sha256"] = ev.sha256_file(dev2)
        base.write_json(meta2, meta)
        self.assertEqual(self.run_classifier("shuffled", ["--skip-evaluation"], dev=dev2, dev_meta=meta2), 0)
        self.assertEqual((self.root / "original/checkpoint.pt").read_bytes(),
                         (self.root / "shuffled/checkpoint.pt").read_bytes())
        self.assertEqual((self.root / "original/predictions.jsonl").read_bytes(),
                         (self.root / "shuffled/predictions.jsonl").read_bytes())

    def test_failed_overfit_gate_stops_before_training(self):
        config = self.write_config(tiny_config(overfit_check={"epochs": 1}), "strict.json")
        self.assertEqual(self.run_classifier("run", config=config), 1)
        gate = json.loads((self.root / "run/overfit_check.json").read_text())
        self.assertFalse(gate["passed"])
        self.assertFalse((self.root / "run/checkpoint.pt").exists())
        record = self.record("run")
        self.assertEqual(record["status"], "overfit_check_failed")
        self.assertEqual(record["encoder"]["frozen"]["trainable_parameters"], 0)

    def test_overfit_check_only_stops_before_training_and_keeps_cache(self):
        self.assertEqual(self.run_classifier("gate", ["--overfit-check-only", "--extract-batch-size", "3"]), 0)
        record = self.record("gate")
        self.assertEqual(record["status"], "overfit_check_only")
        self.assertTrue(record["overfit_check"]["passed"])
        self.assertEqual(record["features"]["extraction_batch_size"], 3)
        self.assertGreater(record["resources"]["extraction_audio_sec_per_sec"], 0)
        self.assertFalse((self.root / "gate/checkpoint.pt").exists())
        self.assertFalse((self.root / "gate/predictions.jsonl").exists())
        # The full run then reuses every cached clip.
        self.assertEqual(self.run_classifier("full", audio=False), 0)
        self.assertEqual(self.record("full")["features"]["extracted_this_run"], {"train": 0, "dev": 0})

    def test_wrong_dev_audio_is_an_explicit_failure(self):
        path = self.root / "audio" / self.dev_rows[0]["audio_path"]
        path.write_bytes(b"not the manifest audio")
        self.assertEqual(self.run_classifier("run"), 2)
        predictions = ev.read_jsonl(self.root / "run/predictions.jsonl")
        self.assertEqual(len(predictions), len(self.dev_rows))
        failed = [p for p in predictions if p["status"] != "ok"]
        self.assertEqual([(p["status"], p["error_code"]) for p in failed],
                         [("audio_error", "audio_integrity_mismatch")])
        self.assertNotIn("scores", failed[0])
        metrics = json.loads((self.root / "run/evaluation/metrics.json").read_text())
        self.assertEqual((metrics["expected"], metrics["failed"]), (20, 1))

    def test_matching_hash_but_undecodable_dev_audio_is_retained(self):
        row = self.dev_rows[0]
        path = self.root / "audio" / row["audio_path"]
        path.write_bytes(b"manifest matches these bytes, but they are not audio")
        row.update(audio_sha256=ev.sha256_file(path), encoded_bytes=path.stat().st_size)
        base.write_jsonl(self.dev, self.dev_rows)
        meta = json.loads(self.dev_meta.read_text())
        meta["manifest_sha256"] = ev.sha256_file(self.dev)
        base.write_json(self.dev_meta, meta)
        self.assertEqual(self.run_classifier("run"), 2)
        predictions = ev.read_jsonl(self.root / "run/predictions.jsonl")
        self.assertEqual(len(predictions), len(self.dev_rows))
        self.assertEqual(predictions[0]["status"], "audio_error")
        self.assertEqual(predictions[0]["error_code"], "audio_unreadable")
        self.assertEqual(sum(p["status"] == "ok" for p in predictions), len(predictions) - 1)
        metrics = json.loads((self.root / "run/evaluation/metrics.json").read_text())
        self.assertEqual((metrics["expected"], metrics["failed"]), (20, 1))

    def test_model_failure_is_isolated_and_other_clips_are_cached(self):
        original = ae.AcousticEncoder.extract
        bad_id = self.dev_rows[0]["clip_id"]

        def extract(encoder, clips, batch_size=8):
            if any(cid == bad_id for cid, _ in clips):
                raise RuntimeError("synthetic clip inference failure")
            return original(encoder, clips, batch_size=batch_size)

        with patch.object(ae.AcousticEncoder, "extract", extract):
            self.assertEqual(self.run_classifier("run"), 2)
        predictions = ev.read_jsonl(self.root / "run/predictions.jsonl")
        self.assertEqual(predictions[0]["status"], "inference_error")
        self.assertEqual(predictions[0]["error_code"], "feature_extraction_failed")
        self.assertEqual(sum(p["status"] == "ok" for p in predictions), len(predictions) - 1)
        self.assertEqual(self.record("run")["features"]["extracted_this_run"]["dev"], 19)
        # A retry extracts only the previously failing clip and recovers full coverage.
        self.assertEqual(self.run_classifier("retry"), 0)
        self.assertEqual(self.record("retry")["features"]["extracted_this_run"], {"train": 0, "dev": 1})
        metrics = json.loads((self.root / "run/evaluation/metrics.json").read_text())
        self.assertEqual((metrics["expected"], metrics["failed"]), (20, 1))

    def test_batch_failure_can_recover_without_dropping_clips(self):
        original = ae.AcousticEncoder.extract

        def extract(encoder, clips, batch_size=8):
            if len(clips) > 1:
                raise RuntimeError("synthetic batch memory limit")
            return original(encoder, clips, batch_size=batch_size)

        with patch.object(ae.AcousticEncoder, "extract", extract):
            self.assertEqual(self.run_classifier("run"), 0)
        self.assertEqual(self.record("run")["features"]["extracted_this_run"], {"train": 40, "dev": 20})
        self.assertEqual(self.record("run")["counts"]["dev_failed"], 0)

    def test_cache_write_error_is_a_setup_error(self):
        with patch.object(ae.FeatureCache, "write", side_effect=OSError("disk full")):
            self.assertEqual(self.run_classifier("run"), 1)
        self.assertFalse((self.root / "run/predictions.jsonl").exists())

    def test_config_rejects_unfrozen_encoder_and_dev_selection(self):
        for section, values, message in (("encoder", {"frozen": False}, "frozen"),
                                         ("optimization", {"early_stopping": True}, "early stopping"),
                                         ("optimization", {"checkpoint_selection": "best_dev"}, "last epoch")):
            path = self.write_config(tiny_config(**{section: values}), "bad.json")
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                clf.load_config(path)


if __name__ == "__main__":
    unittest.main()
