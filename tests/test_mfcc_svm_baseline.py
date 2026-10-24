"""Offline W1-11 checks on synthetic audio; no real-audio results are claimed."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.evaluation import evaluate as ev
from src.experiments import mfcc_svm_baseline as base

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "configs/labels.json"
LABELS = ev.load_labels(LABELS_PATH)
CONFIG = base.load_config(base.DEFAULT_CONFIG)
RATE = 16000


def tone(label_index, variant, seconds=1.0, rate=RATE):
    """Noise with a label-specific spectral peak (a crude 'formant').

    MFCCs describe the spectral envelope, not pitch, so classes differ in envelope.
    """
    rng = np.random.default_rng(1000 * label_index + variant)
    samples = max(1, int(seconds * rate))
    spectrum = np.fft.rfft(rng.standard_normal(samples))
    freqs = np.fft.rfftfreq(samples, 1 / rate)
    centre = 250.0 * 1.15 ** label_index * (1 + rng.uniform(-0.02, 0.02))
    envelope = 0.05 + np.exp(-0.5 * ((freqs - centre) / (0.08 * centre)) ** 2)
    wave = np.fft.irfft(spectrum * envelope, n=samples)
    return (0.2 * wave / (np.abs(wave).max() + 1e-9)).astype(np.float32)


def write_split(root, split, per_label, broken=()):
    audio_root = root / "audio"
    rows = []
    for index, label in enumerate(LABELS):
        for variant in range(per_label):
            clip_id = hashlib.sha256(f"{split}-{label}-{variant}".encode()).hexdigest()[:16]
            path = audio_root / split / f"{clip_id}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            if clip_id in broken or (split, label, variant) in broken:
                path.write_bytes(b"not a wav file")
            else:
                sf.write(path, tone(index, variant + (0 if split == "train" else 500)), RATE,
                         subtype="FLOAT")
            rows.append({"clip_id": clip_id, "audio_path": f"{split}/{clip_id}.wav", "label": label,
                         "split": split, "source_id": None, "speaker_id": None, "domain": "unknown",
                         "duration_sec": 1.0, "sample_rate_hz": RATE})
    manifest = root / f"{split}.jsonl"
    base.write_jsonl(manifest, rows)
    metadata = root / f"{split}.metadata.json"
    base.write_json(metadata, {"schema_version": "1.0", "dataset_version": "fixture",
                               "split_version": "fixture-split", "manifest_sha256": ev.sha256_file(manifest),
                               "labels_sha256": ev.sha256_file(LABELS_PATH),
                               "evaluation_label_sets": {split: list(LABELS)}})
    return manifest, metadata, rows


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = base.MfccExtractor(CONFIG["features"])

    def test_dimension_and_names(self):
        vector = self.extractor.summary(tone(3, 0))
        self.assertEqual(vector.shape, (78,))
        self.assertEqual(len(base.feature_names(CONFIG["features"])), 78)
        self.assertTrue(np.isfinite(vector).all())

    def test_deterministic_and_short_clip(self):
        np.testing.assert_array_equal(self.extractor.summary(tone(2, 1)),
                                      self.extractor.summary(tone(2, 1)))
        short = self.extractor.summary(tone(2, 1, seconds=0.01))  # shorter than one frame
        self.assertTrue(np.isfinite(short).all())

    def test_dct_matches_scipy(self):
        from scipy.fft import dct
        x = np.random.default_rng(0).standard_normal((40, 7))
        np.testing.assert_allclose(base.dct_ii_ortho(40, 40) @ x, dct(x, type=2, norm="ortho", axis=0),
                                   atol=1e-10)

    def test_filterbank_matches_librosa_reference(self):
        try:
            import librosa
        except ImportError:
            self.skipTest("librosa not installed; optional cross-check")
        f = CONFIG["features"]
        expected = librosa.filters.mel(sr=RATE, n_fft=f["n_fft"], n_mels=f["n_mels"], fmin=f["f_min_hz"],
                                       fmax=f["f_max_hz"], htk=True, norm=None)
        np.testing.assert_allclose(self.extractor.filters, expected, atol=1e-6)

    def test_frame_count(self):
        # 1 s at 16 kHz, 400-sample frames every 160 samples -> 1 + (16000 - 400) // 160 frames.
        self.assertEqual(self.extractor.mfcc(tone(1, 0)).shape, (13, 98))

    def test_resampling_and_stereo(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stereo.wav"
            mono = tone(4, 0, rate=8000)
            sf.write(path, np.stack([mono, mono], axis=1), 8000)
            wave, original_rate = base.load_wave(path, RATE)
            self.assertEqual(original_rate, 8000)
            self.assertEqual(wave.size, 2 * mono.size)

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            base.resolve_audio_path("/data", "../secret.wav")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.train_meta, _ = write_split(self.root, "train", per_label=4)

    def tearDown(self):
        self.temp.cleanup()

    def run_baseline(self, dev, dev_meta, name, extra=()):
        argv = ["--train-manifest", str(self.train), "--train-metadata", str(self.train_meta),
                "--dev-manifest", str(dev), "--dev-metadata", str(dev_meta),
                "--audio-root", str(self.root / "audio"), "--labels", str(LABELS_PATH),
                "--output-dir", str(self.root / name), *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return base.main(argv)

    def test_end_to_end_outputs_and_evaluation(self):
        dev, dev_meta, dev_rows = write_split(self.root, "dev", per_label=2)
        self.assertEqual(self.run_baseline(dev, dev_meta, "run"), 0)
        out = self.root / "run"
        for name in ("model.joblib", "predictions.jsonl", "dev_decision_scores.jsonl", "features.npz",
                     "failures.jsonl", "config.json", "run_record.json", "evaluation/metrics.json",
                     "evaluation/confusion_matrix.csv"):
            self.assertTrue((out / name).is_file(), name)
        predictions = ev.read_jsonl(out / "predictions.jsonl")
        self.assertEqual([p["clip_id"] for p in predictions], [r["clip_id"] for r in dev_rows])
        record = json.loads((out / "run_record.json").read_text())
        self.assertEqual(record["result_role"], base.RESULT_ROLE)
        self.assertEqual(record["development_evaluation"]["evaluation_mode"], "complete")
        self.assertEqual(record["development_evaluation"]["expected"], 40)
        # Neighbouring synthetic classes overlap with only 4 training clips each, so we only
        # require clearly-above-chance (5%) accuracy: a label/ID misalignment would score ~5%.
        self.assertGreater(record["development_evaluation"]["accuracy"], 0.3)
        self.assertIn(record["model_selection"]["selected_C"], CONFIG["classifier"]["C_grid"])
        # The saved model reproduces the written predictions from the saved features.
        import joblib
        saved = joblib.load(out / "model.joblib")
        features = np.load(out / "features.npz")
        replay = dict(zip(features["dev_ids"], saved["pipeline"].predict(features["dev_x"])))
        self.assertEqual({p["clip_id"]: p["predicted_label"] for p in predictions}, replay)
        self.assertEqual(saved["labels"], LABELS)
        self.assertEqual(record["artifacts"]["predictions.jsonl"],
                         ev.sha256_file(out / "predictions.jsonl"))

    def test_dev_labels_do_not_influence_predictions(self):
        dev, dev_meta, dev_rows = write_split(self.root, "dev", per_label=2)
        self.assertEqual(self.run_baseline(dev, dev_meta, "original", ["--skip-evaluation"]), 0)
        # Same audio, permuted gold labels: predictions must be byte-identical apart from revision.
        shuffled = [dict(r) for r in dev_rows]
        permuted = list(np.random.default_rng(7).permutation([r["label"] for r in dev_rows]))
        for row, label in zip(shuffled, permuted):
            row["label"] = str(label)
        dev2 = self.root / "dev_shuffled.jsonl"
        base.write_jsonl(dev2, shuffled)
        meta2 = self.root / "dev_shuffled.metadata.json"
        meta = json.loads(dev_meta.read_text())
        meta["manifest_sha256"] = ev.sha256_file(dev2)
        base.write_json(meta2, meta)
        self.assertEqual(self.run_baseline(dev2, meta2, "shuffled", ["--skip-evaluation"]), 0)
        strip = lambda p: [(r["clip_id"], r["predicted_label"]) for r in ev.read_jsonl(p)]
        self.assertEqual(strip(self.root / "original/predictions.jsonl"),
                         strip(self.root / "shuffled/predictions.jsonl"))

    def test_unreadable_dev_clip_is_an_explicit_failure(self):
        dev, dev_meta, dev_rows = write_split(self.root, "dev", per_label=2, broken={("dev", "LEB", 0)})
        self.assertEqual(self.run_baseline(dev, dev_meta, "run"), 2)
        predictions = ev.read_jsonl(self.root / "run/predictions.jsonl")
        self.assertEqual(len(predictions), len(dev_rows))
        failed = [p for p in predictions if p["status"] != "ok"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["status"], "audio_error")
        self.assertIsNone(failed[0]["predicted_label"])
        metrics = json.loads((self.root / "run/evaluation/metrics.json").read_text())
        self.assertEqual(metrics["expected"], 40)
        self.assertEqual(metrics["failed"], 1)

    def test_refuses_existing_output_and_overlapping_ids(self):
        dev, dev_meta, _ = write_split(self.root, "dev", per_label=2)
        (self.root / "exists").mkdir()
        self.assertEqual(self.run_baseline(dev, dev_meta, "exists"), 1)
        self.assertEqual(self.run_baseline(self.train, self.train_meta, "wrong-split"), 1)

    def test_rejects_tampered_manifest(self):
        dev, dev_meta, rows = write_split(self.root, "dev", per_label=2)
        base.write_jsonl(dev, rows[:-1])  # hash no longer matches metadata
        self.assertEqual(self.run_baseline(dev, dev_meta, "run"), 1)


class CommittedPilotTests(unittest.TestCase):
    def test_default_manifests_are_the_frozen_pilot(self):
        for split, count in (("train", 500), ("dev", 200)):
            rows, found, meta = base.load_split(
                ROOT / f"data/manifests/{split}_pilot.jsonl",
                ROOT / f"data/manifests/{split}_pilot.metadata.json", LABELS_PATH, LABELS)
            self.assertEqual((found, len(rows)), (split, count))
            self.assertEqual(meta["split_version"], "w1-07-pilot-v1-e61cd569ac7b")


if __name__ == "__main__":
    unittest.main()
