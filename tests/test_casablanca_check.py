"""Offline checks for the Casablanca check of the Idea 1 heads; tiny random encoder, synthetic audio.

Run in the acoustic environment, from the repository root:
    python -m unittest discover -s tests -p 'test_casablanca_check.py' -v
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import casablanca_check as cc
from src.experiments import layer_mixture as lm
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae
from test_layer_mixture import TINY, tiny_config, tiny_provenance
from test_mfcc_svm_baseline import write_split

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "configs/labels.json"
PRESENT = ["ALG", "EGY", "JOR", "MAU", "MOR", "PAL", "UAE", "YEM"]


def tiny_encoder(feature_config, device=None, cache_dir=None):
    torch.manual_seed(0)
    return ae.AcousticEncoder(feature_config, TINY, device="cpu")


def refuse_encoder(*args, **kwargs):
    raise AssertionError("the encoder must not load when every clip is cached")


def quiet(function, *args, **kwargs):
    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
        return function(*args, **kwargs), stderr.getvalue()


class CasablancaCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # A completed tiny pilot run, exactly as layer_mixture.py writes it.
        train, train_meta, train_rows = write_split(self.root, "train", per_label=3)
        dev, dev_meta, dev_rows = write_split(self.root, "dev", per_label=1)
        self.pilot_args = ["--train-manifest", str(train), "--train-metadata", str(train_meta),
                           "--dev-manifest", str(dev), "--dev-metadata", str(dev_meta)]
        config = self.root / "config.json"
        base.write_json(config, tiny_config())
        encoder = tiny_encoder(clf.feature_config(tiny_config()["encoder"]))
        cache = ae.FeatureCache(self.root / "cache", encoder.provenance())
        for rows in (train_rows, dev_rows):
            cache.write(encoder.extract_manifest(base.without_labels(rows), self.root / "audio"))
        code, err = quiet(lm.main, [*self.pilot_args, "--output-root", str(self.root / "cache"),
                                    "--labels", str(LABELS_PATH), "--config", str(config),
                                    "--output-dir", str(self.root / "pilot")], provenance_fn=tiny_provenance)
        self.assertEqual(code, 0, err)
        # A Casablanca-like split: 8 present labels, one row marked audio_error, uncached audio.
        cas_root = self.root / "cas"
        _, _, rows = write_split(cas_root, "test", per_label=2)
        rows = [dict(r, split="dev", preparation_status="ok") for r in rows if r["label"] in PRESENT]
        # Like the real manifest: the failed row has no decoded audio identity, duration or rate.
        rows[0].update(preparation_status="audio_error", duration_sec=None, sample_rate_hz=None,
                       audio_sha256=None, encoded_bytes=None)
        self.cas_rows = rows
        self.cas_manifest, self.cas_meta = self.write_casablanca(rows, "cas")

    def tearDown(self):
        self.temp.cleanup()

    def write_casablanca(self, rows, name):
        manifest = self.root / f"{name}.jsonl"
        base.write_jsonl(manifest, rows)
        metadata = self.root / f"{name}.metadata.json"
        base.write_json(metadata, {"schema_version": "1.0", "dataset_version": "fixture",
                                   "split_version": "casablanca-fixture", "split": "dev",
                                   "manifest_sha256": ev.sha256_file(manifest),
                                   "labels_sha256": ev.sha256_file(LABELS_PATH), "test_split_used": False,
                                   "evaluation_label_sets": {"dev": PRESENT}})
        return manifest, metadata

    def check(self, name, load_encoder=tiny_encoder, manifest=None, metadata=None, audio=True, run_dir=None):
        argv = [*self.pilot_args, "--run-dir", str(run_dir or self.root / "pilot"),
                "--output-root", str(self.root / "cache"), "--labels", str(LABELS_PATH),
                "--casablanca-manifest", str(manifest or self.cas_manifest),
                "--casablanca-metadata", str(metadata or self.cas_meta), "--output-dir", str(self.root / name)]
        if audio:
            argv += ["--audio-root", str(self.root / "cas" / "audio")]
        return quiet(cc.main, argv, load_encoder=load_encoder, provenance_fn=tiny_provenance)

    def test_end_to_end_with_failures_reproduction_and_evaluator(self):
        code, err = self.check("run")
        self.assertEqual(code, 0, err)
        out = self.root / "run"
        record = json.loads((out / "run_record.json").read_text())
        self.assertTrue(record["features"]["extraction"]["encoder_loaded"])
        self.assertEqual(record["counts"]["failures"], 1)
        reproduction = json.loads((out / "reproduction.json").read_text())["heads"]
        self.assertEqual(len(reproduction), 5 * 2)  # headline arms x seeds
        self.assertTrue(all(h["same_predictions"] and h["max_abs_logit_difference"] <= cc.REPRODUCTION_TOLERANCE
                            for h in reproduction.values()))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual([a["arm"] for a in summary["arms"]], list(cc.HEADLINE))
        for arm in cc.HEADLINE:
            folder = out / "arms" / arm / "seed_1"
            predictions = ev.read_jsonl(folder / "predictions.jsonl")
            self.assertEqual([p["clip_id"] for p in predictions], [r["clip_id"] for r in self.cas_rows])
            self.assertEqual(predictions[0]["status"], "audio_error")
            self.assertTrue(all(len(p["scores"]) == 20 for p in predictions[1:]))  # never masked to 8 labels
            metrics = json.loads((folder / "evaluation/metrics.json").read_text())
            self.assertEqual(metrics["expected"], len(self.cas_rows))
            self.assertEqual(metrics["macro_f1_labels"], PRESENT)

    def test_rerun_uses_the_cache_and_is_identical(self):
        self.assertEqual(self.check("first")[0], 0)
        code, err = self.check("second", load_encoder=refuse_encoder, audio=False)
        self.assertEqual(code, 0, err)
        for path in ("arms/mixture_all/seed_2/predictions.jsonl", "arms/b0_final/seed_1/predictions.jsonl"):
            self.assertEqual((self.root / "first" / path).read_bytes(), (self.root / "second" / path).read_bytes())

    def test_casablanca_labels_never_change_predictions(self):
        self.assertEqual(self.check("original")[0], 0)
        shuffled = [dict(r, label=self.cas_rows[(i + 5) % len(self.cas_rows)]["label"])
                    for i, r in enumerate(self.cas_rows)]
        manifest, metadata = self.write_casablanca(shuffled, "shuffled")
        code, err = self.check("shuffled", manifest=manifest, metadata=metadata)
        self.assertEqual(code, 0, err)
        path = "arms/selected_layer/seed_1/predictions.jsonl"
        self.assertEqual((self.root / "original" / path).read_bytes(), (self.root / "shuffled" / path).read_bytes())

    def test_changed_head_fails_the_reproduction_check(self):
        run_dir = self.root / "pilot_copy"
        shutil.copytree(self.root / "pilot", run_dir)
        head = run_dir / "arms/mixture_all/seed_1/head.pt"
        saved = torch.load(head, weights_only=True)
        saved["state_dict"]["linear.weight"] *= 1.5
        torch.save(saved, head)
        code, err = self.check("run", run_dir=run_dir)
        self.assertEqual(code, 1)
        self.assertIn("does not reproduce", err)

    def test_uncached_clips_need_the_audio_root(self):
        code, err = self.check("run", audio=False)
        self.assertEqual(code, 1)
        self.assertIn("--audio-root", err)
        self.assertFalse((self.root / "run").exists())

    def test_test_split_is_refused(self):
        metadata = json.loads(self.cas_meta.read_text())
        metadata["test_split_used"] = True
        path = self.root / "test_meta.json"
        base.write_json(path, metadata)
        code, err = self.check("run", metadata=path)
        self.assertEqual(code, 1)
        self.assertIn("validation", err)


if __name__ == "__main__":
    unittest.main()
