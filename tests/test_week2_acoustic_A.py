"""Architecture A contract, scale and long-input regression checks (offline)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.experiments import acoustic_classifier as clf
from src.experiments import acoustic_A_predict as inference
from src.models import acoustic_encoder as ae
import test_acoustic_classifier as pilot

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_full_config_accepts_only_the_approved_release(self):
        config = clf.load_config(ROOT / "configs/week2_acoustic_A.json")
        rows, metas = [], []
        for split in ("train", "dev"):
            path = ROOT / f"src/data/manifests/{split}.jsonl"
            meta = ROOT / f"src/data/manifests/{split}.metadata.json"
            r, _, m = clf.base.load_split(path, meta, ROOT / "configs/labels.json",
                                         clf.ev.load_labels(ROOT / "configs/labels.json"))
            rows.append(r)
            metas.append(m)
        clf.validate_experiment_data(config, *rows, *metas)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            clf.validate_experiment_data(config, rows[0][:500], rows[1], *metas)
        changed = {**metas[1], "split_version": "pilot"}
        with self.assertRaisesRegex(ValueError, "split_version"):
            clf.validate_experiment_data(config, *rows, metas[0], changed)
        self.assertEqual(config["encoder"]["cached_layers"], [25])
        self.assertEqual(config["optimization"]["epochs"], 30)

    def test_similarity_diagnostic_is_bounded_at_full_scale(self):
        # 67k pairwise entries would require >17 GB even with only 4 feature dimensions.
        matrix = np.random.default_rng(1).standard_normal((67015, 4)).astype(np.float32)
        stats = clf.feature_stats(matrix)
        self.assertEqual(stats["clips"], 67015)
        self.assertEqual(stats["cosine_sample_clips"], 512)
        self.assertTrue(stats["finite"])

    def test_preflight_uses_longest_inputs_and_preserves_failures(self):
        class FakeEncoder:
            device = torch.device("cpu")
            config = ae.FeatureConfig()

            def extract(self, clips, batch_size):
                if clips[0][0] == "long-dev":
                    raise RuntimeError("simulated out of memory")
                return ae.ClipFeatures([clips[0][0]], np.zeros((1, 1, 848)), np.array([100]), ["final"])

        def row(cid, duration):
            return {"clip_id": cid, "duration_sec": duration, "audio_path": f"{cid}.wav"}

        train = [row("short", 1.0), row("long-train", 100.0)]
        dev = [row("long-dev", 212.38), row("short-dev", 2.0)]
        with patch.object(clf.base, "verify_audio"), patch.object(ae, "load_waveform", return_value=torch.zeros(10)):
            result = clf.long_clip_preflight(FakeEncoder(), train, dev, Path("/audio"))
        self.assertFalse(result["passed"])
        self.assertEqual([c["clip_id"] for c in result["checks"]], ["long-train", "long-dev"])
        self.assertEqual(result["checks"][1]["status"], "failed")


class PredictionTests(unittest.TestCase):
    def test_checkpoint_prediction_reproduces_written_logits_without_training(self):
        fixture = pilot.PipelineTests(methodName="test_end_to_end_outputs_scores_and_frozen_encoder")
        fixture.setUp()
        try:
            self.assertEqual(fixture.run_classifier("train-run"), 0)
            checkpoint = fixture.root / "train-run/checkpoint.pt"
            saved = torch.load(checkpoint, weights_only=True)
            # The offline tiny encoder has 32 dimensions; no actual 300M training is implied.
            saved["config"]["experiment"] = {"architecture": "A"}
            torch.save(saved, checkpoint)
            out = fixture.root / "predict-run"
            with contextlib.redirect_stdout(io.StringIO()), patch.object(clf, "train_head", side_effect=AssertionError("inference fitted a head")):
                code = inference.main(["--checkpoint", str(checkpoint), "--manifest", str(fixture.dev),
                    "--metadata", str(fixture.dev_meta), "--output-root", str(fixture.root / "cache"),
                    "--device", "cpu", "--output-dir", str(out)], load_encoder=pilot.tiny_encoder)
            self.assertEqual(code, 0)
            original = clf.ev.read_jsonl(fixture.root / "train-run/predictions.jsonl")
            repeated = clf.ev.read_jsonl(out / "predictions.jsonl")
            np.testing.assert_allclose([r["scores"] for r in repeated], [r["scores"] for r in original], atol=1e-6, rtol=0)
            self.assertFalse(json.loads((out / "prediction_record.json").read_text())["fit_performed"])
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
