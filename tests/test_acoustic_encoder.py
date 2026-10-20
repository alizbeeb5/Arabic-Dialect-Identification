"""Acoustic feature interface checks on a tiny random encoder (no checkpoint download).

Run in the acoustic environment:
    python -m unittest discover -s tests -p 'test_acoustic_encoder.py' -v
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))
import acoustic_encoder as ae  # noqa: E402

TINY = ae.Architecture(d_model=32, nhead=4, num_layers=2, d_ffn=64)
TOLERANCE = 1e-5
NUM_LABELS = 20


def tiny_encoder(**config):
    torch.manual_seed(0)
    return ae.AcousticEncoder(ae.FeatureConfig(**config), TINY, device="cpu")


def synthetic_clips(seconds=(0.7, 2.3, 1.1, 3.0, 0.3)):
    generator = torch.Generator().manual_seed(1)
    return [(f"clip_{i:03d}", 0.1 * torch.randn(int(s * 16000), generator=generator))
            for i, s in enumerate(seconds)]


class ClassifierHead(torch.nn.Module):
    """Learned layer mixture plus a 20-way linear classifier."""

    def __init__(self, layers, dim):
        super().__init__()
        self.layer_weights = torch.nn.Parameter(torch.zeros(layers))
        self.linear = torch.nn.Linear(dim, NUM_LABELS)

    def forward(self, features):  # [B, L, D] -> [B, 20]
        mixed = (features * self.layer_weights.softmax(0)[None, :, None]).sum(1)
        return self.linear(mixed)


class VerifierHead(torch.nn.Module):
    """Scores a (clip, claimed dialect) pair from the same pooled features."""

    def __init__(self, layers, dim, embed=16):
        super().__init__()
        self.layer_weights = torch.nn.Parameter(torch.zeros(layers))
        self.project = torch.nn.Linear(dim, embed)
        self.dialects = torch.nn.Embedding(NUM_LABELS, embed)

    def forward(self, features, claimed):  # [B, L, D], [B] -> [B]
        mixed = (features * self.layer_weights.softmax(0)[None, :, None]).sum(1)
        return torch.nn.functional.cosine_similarity(self.project(mixed), self.dialects(claimed), dim=-1)


class PoolingTests(unittest.TestCase):
    def test_mean_pool_ignores_padding_values(self):
        hidden = torch.randn(3, 2, 6, 4)
        mask = ae.frame_mask(torch.tensor([6, 2]), 6)
        garbage = hidden.clone()
        garbage[:, 1, 2:] = 1e6
        expected = hidden[:, 1, :2].mean(dim=1)
        self.assertTrue(torch.allclose(ae.masked_mean_pool(garbage, mask)[:, 1], expected))

    def test_mean_std_pool_ignores_padding_values(self):
        hidden = torch.randn(2, 5, 3)
        mask = ae.frame_mask(torch.tensor([5, 3]), 5)
        garbage = hidden.clone()
        garbage[1, 3:] = -1e6
        pooled = ae.masked_mean_std_pool(garbage, mask)
        real = hidden[1, :3]
        self.assertTrue(torch.allclose(pooled[1, :3], real.mean(0), atol=1e-6))
        self.assertTrue(torch.allclose(pooled[1, 3:], real.std(0, unbiased=False), atol=1e-5))


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = tiny_encoder()
        cls.clips = synthetic_clips()

    def test_features_are_finite_with_expected_dimensions(self):
        result = self.encoder.extract(self.clips, batch_size=2)
        self.assertEqual(result.features.shape, (len(self.clips), TINY.num_representations, TINY.d_model))
        self.assertEqual(result.features.dtype, np.float32)
        self.assertTrue(np.isfinite(result.features).all())
        self.assertEqual(result.layer_names[0], "frontend")
        self.assertEqual(result.layer_names[-1], "final")
        # 40 ms encoder frames: ceil((samples // 160 + 1) / 4).
        expected = [-(-(wave.numel() // 160 + 1) // 4) for _, wave in self.clips]
        self.assertEqual(result.n_frames.tolist(), expected)

    def test_clip_ids_follow_input_order_and_values_follow_clips(self):
        reference = self.encoder.extract(self.clips, batch_size=2)
        shuffled = self.clips[:]
        random.Random(3).shuffle(shuffled)
        result = self.encoder.extract(shuffled, batch_size=3)
        self.assertEqual(result.clip_ids, [clip_id for clip_id, _ in shuffled])
        lookup = dict(zip(reference.clip_ids, reference.features))
        for clip_id, features in zip(result.clip_ids, result.features):
            np.testing.assert_allclose(features, lookup[clip_id], atol=TOLERANCE, rtol=0)

    def test_duplicate_clip_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.encoder.extract([self.clips[0], self.clips[0]])

    def test_batched_clip_matches_clip_encoded_alone(self):
        short, long_ = self.clips[4], self.clips[3]
        alone = self.encoder.extract([short]).features[0]
        batched = self.encoder.extract([short, long_], batch_size=2).features[0]
        np.testing.assert_allclose(batched, alone, atol=TOLERANCE, rtol=0)

    def test_repeated_inference_is_reproducible(self):
        first = self.encoder.extract(self.clips, batch_size=2).features
        second = self.encoder.extract(self.clips, batch_size=2).features
        self.assertLessEqual(float(np.abs(first - second).max()), TOLERANCE)
        self.assertFalse(self.encoder.training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))

    def test_layer_selection_and_mean_std_pooling(self):
        encoder = tiny_encoder(layers=(0, 3), pooling="mean_std")
        result = encoder.extract(self.clips[:2])
        self.assertEqual(result.features.shape, (2, 2, 2 * TINY.d_model))
        self.assertEqual(result.layer_names, ["frontend", "final"])
        with self.assertRaises(ValueError):
            ae.FeatureConfig(layers=(99,)).layer_indices(TINY)

    def test_both_heads_consume_the_same_pooled_features(self):
        result = self.encoder.extract(self.clips)
        features = torch.from_numpy(result.features)
        layers, dim = features.shape[1:]
        classifier, verifier = ClassifierHead(layers, dim), VerifierHead(layers, dim)
        logits = classifier(features)
        scores = verifier(features, torch.arange(len(self.clips)) % NUM_LABELS)
        self.assertEqual(logits.shape, (len(self.clips), NUM_LABELS))
        self.assertEqual(scores.shape, (len(self.clips),))
        (logits.sum() + scores.sum()).backward()
        self.assertIsNotNone(classifier.linear.weight.grad)
        self.assertIsNotNone(verifier.project.weight.grad)


class AudioTests(unittest.TestCase):
    def test_waveform_is_mono_resampled_and_cropped(self):
        config = ae.FeatureConfig(max_duration_sec=0.5)
        stereo = torch.randn(2, 8000)
        wave = ae.prepare_waveform(stereo, 8000, config)
        self.assertEqual(wave.shape, (8000,))  # 1 s at 8 kHz -> 16 kHz, cropped to 0.5 s

    def test_audio_path_must_stay_under_audio_root(self):
        for bad in ("../outside.wav", "/abs/clip.wav"):
            with self.assertRaises(ValueError):
                ae.resolve_audio_path("/data/audio", bad)

    def test_config_hash_tracks_feature_settings(self):
        self.assertEqual(ae.FeatureConfig().sha256(), ae.FeatureConfig().sha256())
        self.assertNotEqual(ae.FeatureConfig().sha256(), ae.FeatureConfig(pooling="mean_std").sha256())
        self.assertNotEqual(ae.FeatureConfig().sha256(), ae.FeatureConfig(max_duration_sec=30.0).sha256())


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.encoder = tiny_encoder()
        audio_root = self.root / "audio"
        audio_root.mkdir()
        self.rows = []
        for clip_id, wave in synthetic_clips((0.4, 1.2, 0.8)):
            sf.write(audio_root / f"{clip_id}.wav", wave.numpy(), 16000, subtype="PCM_16")
            self.rows.append({"clip_id": clip_id, "audio_path": f"{clip_id}.wav",
                              "audio_sha256": ae.sha256_file(audio_root / f"{clip_id}.wav")})
        self.audio_root = audio_root

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_preserves_values_and_requested_order(self):
        result = self.encoder.extract_manifest(self.rows, self.audio_root)
        cache = ae.FeatureCache(self.root / "out", self.encoder.provenance())
        self.assertEqual(cache.write(result), 3)
        self.assertEqual(cache.write(result), 0)  # already cached, same keys
        order = [row["clip_id"] for row in reversed(self.rows)]
        loaded = cache.load(order)
        self.assertEqual(loaded.clip_ids, order)
        np.testing.assert_array_equal(loaded.features, result.features[::-1])
        self.assertEqual(cache.missing(self.rows), [])
        provenance = json.loads((cache.root / "config.json").read_text())
        self.assertEqual(provenance["feature_config_sha256"], self.encoder.config.sha256())

    def test_cache_key_changes_with_audio_and_configuration(self):
        provenance = self.encoder.provenance()
        key = ae.clip_cache_key("clip_000", "a" * 64, provenance)
        self.assertNotEqual(key, ae.clip_cache_key("clip_000", "b" * 64, provenance))
        other = tiny_encoder(pooling="mean_std").provenance()
        self.assertNotEqual(key, ae.clip_cache_key("clip_000", "a" * 64, other))
        self.assertNotEqual(ae.FeatureCache(self.root, provenance).root, ae.FeatureCache(self.root, other).root)

    def test_changed_audio_is_reported_missing_and_not_overwritten(self):
        result = self.encoder.extract_manifest(self.rows, self.audio_root)
        cache = ae.FeatureCache(self.root / "out", self.encoder.provenance())
        cache.write(result)
        changed = dict(self.rows[0], audio_sha256="0" * 64)
        self.assertEqual(cache.missing([changed]), [changed])
        result.audio_sha256[0] = "0" * 64
        with self.assertRaises(ValueError):
            cache.write(result)

    def test_manifest_hash_mismatch_is_rejected(self):
        bad = [dict(self.rows[0], audio_sha256="f" * 64)]
        with self.assertRaises(ValueError):
            self.encoder.extract_manifest(bad, self.audio_root)


if __name__ == "__main__":
    unittest.main()
