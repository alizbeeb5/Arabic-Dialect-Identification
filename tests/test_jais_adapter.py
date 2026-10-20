"""Offline contract/resume checks. Fake model calls never claim GPU validation."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.models import jais_adapter as jais
from src.evaluation.evaluate import load_labels, validate_prediction_fields

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    revision = jais.MODEL_REVISION

    def __init__(self, mode="ok", interrupt_at=None):
        self.identity = {"revision": self.revision, "dtype": "offline-fake", "tokens": 128}
        self.calls, self.mode, self.interrupt_at = [], mode, interrupt_at

    def predict(self, ids, waveforms, rates):
        self.calls.append(list(ids))
        if self.interrupt_at == len(self.calls):
            raise KeyboardInterrupt("simulated disconnect")
        if self.mode == "raise":
            raise RuntimeError("simulated generation failure")
        if self.mode == "wrong_count":
            return []
        records = [jais.parsed_record(
            clip_id, self.revision, "[msa] synthetic",
            SimpleNamespace(text="synthetic", language="msa", refused=False, truncated=False),
        ) for clip_id in ids]
        if self.mode == "wrong_id":
            records[0]["prediction"]["clip_id"] = "wrong"
        return records


def decoder(payload):
    if payload == b"broken":
        raise ValueError("simulated corrupt audio")
    return np.array([len(payload)], dtype=np.float32), 16000


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio, self.cache = self.root / "audio", self.root / "cache"
        self.audio.mkdir()
        self.clips = []
        for i in range(3):
            (self.audio / f"{i}.wav").write_bytes(f"synthetic-{i}".encode())
            self.clips.append({"clip_id": f"clip-{i}", "audio_path": f"{i}.wav",
                               "label": "LEB", "source_id": "must-not-enter-model"})

    def tearDown(self):
        self.temp.cleanup()

    def run_adapter(self, clips=None, backend=None, **kwargs):
        return jais.run_cached_jais(
            self.clips if clips is None else clips, self.audio, self.cache,
            Backend() if backend is None else backend, decoder=decoder, **kwargs,
        )

    def test_resume_preserves_order_and_skips_backend(self):
        backend = Backend()
        first, cold = self.run_adapter(backend=backend)
        reordered = self.clips[::-1]
        second, warm = self.run_adapter(reordered, backend)
        self.assertEqual(cold["new_outputs"], 3)
        self.assertEqual(warm["cache_hits"], 3)
        self.assertEqual(warm["model_batches"], 0)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(second, first[::-1])

    def test_audio_and_generation_config_changes_invalidate_cache(self):
        self.run_adapter()
        (self.audio / "0.wav").write_bytes(b"changed-audio")
        _, stats = self.run_adapter()
        self.assertEqual((stats["cache_hits"], stats["new_outputs"]), (2, 1))
        changed = Backend()
        changed.identity["tokens"] = 256
        _, stats = self.run_adapter(backend=changed)
        self.assertEqual((stats["cache_hits"], stats["new_outputs"]), (0, 3))

    def test_metadata_and_storage_root_do_not_enter_model_or_cache_identity(self):
        backend = Backend()
        records, _ = self.run_adapter(backend=backend)
        for row in self.clips:
            row["label"], row["source_id"] = "MSA", "changed-metadata"
        renamed = self.audio / "renamed.wav"
        renamed.write_bytes((self.audio / "0.wav").read_bytes())
        self.clips[0]["audio_path"] = renamed.name
        again, stats = self.run_adapter(backend=backend)
        self.assertEqual(stats["cache_hits"], 3)
        self.assertEqual(records, again)
        self.assertEqual(backend.calls, [["clip-0", "clip-1"], ["clip-2"]])

    def test_interruption_reuses_completed_batch(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_adapter(backend=Backend(interrupt_at=2))
        backend = Backend()
        records, stats = self.run_adapter(backend=backend)
        self.assertEqual((stats["cache_hits"], stats["new_outputs"]), (2, 1))
        self.assertEqual(backend.calls, [["clip-2"]])
        self.assertEqual([r["prediction"]["clip_id"] for r in records],
                         [r["clip_id"] for r in self.clips])

    def test_missing_and_corrupt_audio_remain_in_output(self):
        (self.audio / "1.wav").unlink()
        (self.audio / "2.wav").write_bytes(b"broken")
        records, stats = self.run_adapter()
        self.assertEqual((stats["expected"], stats["successful"], stats["failed"]), (3, 1, 2))
        self.assertEqual([r["prediction"]["status"] for r in records],
                         ["ok", "audio_error", "audio_error"])
        self.assertIsNone(records[1]["prediction"]["predicted_label"])
        self.assertEqual(records[2]["prediction"]["error_code"], "decode_failed")

    def test_generation_and_alignment_failures_are_explicit_and_retryable(self):
        for mode in ["raise", "wrong_count", "wrong_id"]:
            with self.subTest(mode=mode):
                records, stats = self.run_adapter(backend=Backend(mode))
                self.assertEqual(stats["failed"], 3)
                self.assertEqual(len(records), 3)
                self.assertFalse(self.cache.exists())
        records, stats = self.run_adapter()
        self.assertEqual((stats["cache_hits"], stats["successful"]), (0, 3))

    def test_tampered_and_malformed_cache_rejected(self):
        self.run_adapter()
        receipt_path = next(self.cache.glob("*.json"))
        receipt = jais.read_json(receipt_path)
        receipt["record"]["transcript"] = "altered"
        jais.atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "Invalid cache receipt"):
            self.run_adapter()
        receipt_path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid cache receipt"):
            self.run_adapter()

    def test_duplicate_ids_and_unsafe_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_adapter([self.clips[0], self.clips[0]])
        for value in ["../outside.wav", str(self.audio / "0.wav"), ""]:
            with self.subTest(path=value), self.assertRaises(ValueError):
                self.run_adapter([dict(self.clips[0], audio_path=value)])
        outside = self.root / "outside.wav"
        outside.write_bytes(b"outside")
        (self.audio / "link.wav").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.run_adapter([dict(self.clips[0], audio_path="link.wav")])

    def test_refusals_unknown_tags_and_truncation(self):
        cases = [("en", False, False, None, "inference_error"),
                 ("msa", True, False, None, "inference_error"),
                 ("LB", False, True, "LEB", "ok")]
        for native, refused, truncated, label, status in cases:
            with self.subTest(native=native, refused=refused):
                item = SimpleNamespace(text="", language=native, refused=refused, truncated=truncated)
                record = jais.parsed_record("test", jais.MODEL_REVISION, "raw", item)
                self.assertEqual(record["prediction"]["predicted_label"], label)
                self.assertEqual(record["prediction"]["status"], status)
                self.assertEqual(record["truncated"], truncated)
                self.assertNotIn("scores", record["prediction"])
                jais.validate_record(record, "test", jais.MODEL_REVISION)

    def test_predictions_accepted_by_shared_validator_and_exported_without_scores(self):
        records, stats = self.run_adapter()
        predictions = [r["prediction"] for r in records]
        validated = validate_prediction_fields(predictions, load_labels(ROOT / "configs/labels.json"))
        self.assertEqual(len(validated), 3)
        output = self.root / "run"
        jais.write_outputs(output, records, {"counts": stats})
        written = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(written, predictions)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            jais.write_outputs(output, records, {})


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def is_floating_point(self):
        return np.issubdtype(self.array.dtype, np.floating)

    def to(self, *args, dtype=None):
        return Tensor(self.array.astype(dtype)) if dtype else self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.array.tolist()


class Features(dict):
    def to(self, device):
        return self


class BackendContractTests(unittest.TestCase):
    def make_adapter(self, parse_error=False):
        device = SimpleNamespace(type="cpu")
        parameter = SimpleNamespace(device=device, dtype="float16")
        model = SimpleNamespace(parameters=lambda: iter([parameter]),
                                generation_config=SimpleNamespace(to_dict=lambda: {}))
        model.eval = lambda: model
        def generate(**kwargs):
            self.generate_args = kwargs
            return Tensor([[1, 2, 10], [1, 2, 20]])
        model.generate = generate

        def process(**kwargs):
            self.processor_args = kwargs
            return Features(input_ids=Tensor([[1, 2], [1, 2]]),
                            audio_features=Tensor(np.ones((2, 3), dtype=np.float32)))
        class Processor:
            tokenizer = SimpleNamespace(
                padding_side="right", apply_chat_template=lambda *a, **k: "audio-only-prompt",
                batch_decode=lambda *a, **k: ["[msa] raw", "[LB] raw"],
            )
            def __call__(self, **kwargs):
                return process(**kwargs)
            def post_process_transcription(self, *args, **kwargs):
                if parse_error:
                    raise ValueError("simulated parser failure")
                return [SimpleNamespace(text="a", language="msa", refused=False, truncated=False),
                        SimpleNamespace(text="b", language="LB", refused=False, truncated=True)]
        fake_torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None), inference_mode=contextlib.nullcontext,
            is_tensor=lambda value: isinstance(value, Tensor),
        )
        return model, Processor(), fake_torch

    def test_backend_preserves_token_types_uses_automatic_detection_and_retains_raw(self):
        model, processor, fake_torch = self.make_adapter()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            adapter = jais.JaisAdapter(model, processor)
            records = adapter.predict(["a", "b"], [np.ones(4), np.ones(8)], [8000, 16000])
        self.assertEqual(self.processor_args["sampling_rate"], [8000, 16000])
        self.assertEqual(self.generate_args["audio_features"].array.dtype, np.float16)
        self.assertTrue(np.issubdtype(self.generate_args["input_ids"].array.dtype, np.integer))
        self.assertFalse(self.generate_args["do_sample"])
        self.assertNotIn("language", self.generate_args)
        self.assertEqual([r["prediction"]["predicted_label"] for r in records], ["MSA", "LEB"])
        self.assertEqual(records[1]["raw_token_ids"], [20])
        self.assertTrue(records[1]["truncated"])

    def test_parser_failure_keeps_generated_text_and_tokens(self):
        model, processor, fake_torch = self.make_adapter(parse_error=True)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            records = jais.JaisAdapter(model, processor).predict(
                ["a", "b"], [np.ones(4), np.ones(8)], [16000, 16000])
        self.assertEqual([r["raw_output"] for r in records], ["[msa] raw", "[LB] raw"])
        self.assertEqual([r["prediction"]["error_code"] for r in records], ["parse_failed"] * 2)
        self.assertEqual(records[0]["raw_token_ids"], [10])


if __name__ == "__main__":
    unittest.main()
