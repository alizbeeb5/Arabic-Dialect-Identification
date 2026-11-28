"""Offline checks for the Idea 1 layer-mixture pilot; tiny random encoder, synthetic audio, no real results.

Run in the acoustic environment, from the repository root:
    python -m unittest discover -s tests -p 'test_layer_mixture.py' -v
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch

from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import layer_mixture as lm
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae
from test_mfcc_svm_baseline import write_split

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "configs/labels.json"
LABELS = ev.load_labels(LABELS_PATH)
TINY = ae.Architecture(d_model=32, nhead=4, num_layers=2, d_ffn=64)  # 4 cached layers
NO_CHECKPOINT = {"repo_id": None, "revision": None, "files": {}}
OPT = lm.load_config(lm.DEFAULT_CONFIG)["optimization"]


def tiny_provenance(feature_config):
    return lm.expected_provenance(feature_config, TINY, NO_CHECKPOINT)


def tiny_config():
    config = lm.load_config(lm.DEFAULT_CONFIG)
    config["encoder"]["feature_dim"] = TINY.d_model
    config["seeds"] = [1, 2]
    config["selection"].update(cv_folds=3, candidate_epochs=[2, 6])
    config["mixtures"] = {"mixture_all": "all", "mixture_middle": [1, 2]}
    return config


def separable(n_per_label=4, dim=32, layers=None, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(20), n_per_label)
    centres = rng.standard_normal((20, dim)) * 3
    x = centres[y] + 0.1 * rng.standard_normal((len(y), dim))
    if layers:
        x = np.repeat(x[:, None, :], layers, axis=1)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


class ProvenanceTests(unittest.TestCase):
    def test_matches_encoder_provenance_without_loading_a_checkpoint(self):
        config = clf.feature_config(tiny_config()["encoder"])
        torch.manual_seed(0)
        encoder = ae.AcousticEncoder(config, TINY, device="cpu")
        expected = json.loads(json.dumps(encoder.provenance()))
        self.assertEqual(json.loads(json.dumps(tiny_provenance(config))), expected)

    def test_pinned_provenance_points_at_the_w1_12_cache_folder(self):
        provenance = lm.expected_provenance(clf.feature_config(lm.load_config(lm.DEFAULT_CONFIG)["encoder"]))
        self.assertEqual(ae.FeatureCache("/x", provenance).root.name, "e4b3c287854e40f9")
        self.assertEqual(len(provenance["layer_names"]), 26)
        self.assertEqual(provenance["checkpoint"]["files"], ae.CHECKPOINT_SHA256)


class SameTestingPipelineAsW112Tests(unittest.TestCase):
    """The pilot is scored exactly like the committed W1-12 acoustic run (and the other pilots)."""
    W1_12 = ROOT / "reports/acoustic_classifier_pilot"

    def test_default_split_and_labels_are_the_frozen_pilot(self):
        args = lm.parse_args(["--output-root", "/x", "--output-dir", "/y"])
        freeze = json.loads((ROOT / "data/manifests/split_freeze.json").read_text())
        record = json.loads((self.W1_12 / "run_record.json").read_text())["data"]
        self.assertEqual(ev.sha256_file(args.train_manifest), freeze["train_pilot_sha256"])
        self.assertEqual(ev.sha256_file(args.dev_manifest), freeze["dev_pilot_sha256"])
        self.assertEqual(ev.sha256_file(args.train_manifest), record["train_manifest_sha256"])
        self.assertEqual(ev.sha256_file(args.dev_manifest), record["dev_manifest_sha256"])
        self.assertEqual(ev.sha256_file(args.labels), record["labels_sha256"])
        metrics = json.loads((self.W1_12 / "metrics.json").read_text())
        self.assertEqual(ev.sha256_file(args.dev_metadata), metrics["metadata_sha256"])

    def test_reads_the_same_cache_entries_as_w1_12(self):
        provenance = lm.expected_provenance(clf.feature_config(lm.load_config(lm.DEFAULT_CONFIG)["encoder"]))
        record = json.loads((self.W1_12 / "run_record.json").read_text())["features"]

        def keys_sha256(path):
            rows = ev.read_jsonl(path)
            return base.sha256_text(base.canonical_json(
                [[r["clip_id"], ae.clip_cache_key(r["clip_id"], r["audio_sha256"], provenance)] for r in rows]))

        self.assertEqual(provenance["feature_config_sha256"], record["feature_config_sha256"])
        self.assertEqual(base.sha256_text(base.canonical_json(provenance)), record["provenance_sha256"])
        self.assertEqual(keys_sha256(ROOT / "data/manifests/train_pilot.jsonl"), record["train_cache_keys_sha256"])
        self.assertEqual(keys_sha256(ROOT / "data/manifests/dev_pilot.jsonl"), record["dev_cache_keys_sha256"])

    def test_scored_by_the_same_unchanged_evaluator(self):
        metrics = json.loads((self.W1_12 / "metrics.json").read_text())
        self.assertEqual(ev.sha256_file(ROOT / "src/evaluation/evaluate.py"), metrics["evaluator_revision"])

    def test_evaluator_command_matches_w1_12(self):
        args = lm.parse_args(["--output-root", "/x", "--output-dir", "/y"])
        commands = []

        def capture(command, **kwargs):
            commands.append(command)
            raise RuntimeError("captured")

        with unittest.mock.patch("subprocess.run", side_effect=capture):
            for call in (lambda: clf.run_evaluator(args, Path("p.jsonl"), Path("out")),
                         lambda: lm.run_evaluator(args, Path("p.jsonl"), Path("out"), "name")):
                with self.assertRaises(RuntimeError):
                    call()
        without_name = [c[:c.index("--model-name")] + c[c.index("--model-name") + 2:] for c in commands]
        self.assertEqual(without_name[0], without_name[1])

    def test_prediction_rows_match_w1_12_format(self):
        old = ev.read_jsonl(self.W1_12 / "predictions.jsonl")
        dev_rows = ev.read_jsonl(ROOT / "data/manifests/dev_pilot.jsonl")
        logits = np.zeros((len(dev_rows), 20))
        logits[:, 3] = 1.0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            lm.write_predictions(path, dev_rows, logits, LABELS, "revision")
            new = ev.read_jsonl(path)
        self.assertEqual(sorted(new[0]), sorted(old[0]))
        self.assertEqual([r["clip_id"] for r in new], [r["clip_id"] for r in old])
        self.assertEqual(new[0]["score_labels"], old[0]["score_labels"])
        self.assertEqual({r["score_type"] for r in new}, {r["score_type"] for r in old})
        self.assertEqual({r["predicted_label"] for r in new}, {LABELS[3]})


class FoldTests(unittest.TestCase):
    def rows(self):
        rows = [{"clip_id": f"{label}-{i}", "label": label, "source_id": None, "audio_sha256": f"h-{label}-{i}"}
                for label in LABELS for i in range(5)]
        rows[0]["source_id"] = rows[1]["source_id"] = rows[2]["source_id"] = "recording-A"
        return rows

    def test_groups_stay_together_and_labels_spread(self):
        rows = self.rows()
        folds, group_of = lm.make_folds(rows, 3, seed=7)
        self.assertEqual(len({int(folds[i]) for i in range(3)}), 1)  # one recording, one fold
        self.assertEqual(group_of[rows[0]["clip_id"]], group_of[rows[2]["clip_id"]])
        for label in LABELS[1:]:
            counts = np.bincount([f for f, r in zip(folds, rows) if r["label"] == label], minlength=3)
            self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_deterministic_and_order_independent(self):
        rows = self.rows()
        first, _ = lm.make_folds(rows, 3, seed=7)
        again, _ = lm.make_folds(list(reversed(rows)), 3, seed=7)
        self.assertEqual(first.tolist(), list(reversed(again.tolist())))

    def test_group_with_two_labels_is_rejected(self):
        rows = self.rows()
        rows[5]["source_id"] = "recording-A"  # label index 1 joins a label-0 recording
        with self.assertRaises(ValueError):
            lm.make_folds(rows, 3, seed=7)


class HeadTests(unittest.TestCase):
    def test_snapshot_during_longer_run_equals_shorter_run(self):
        for layers in (None, 3):
            x, y = separable(layers=layers)
            snapshots = {}
            lm.fit(x, y, OPT, 20, seed=4, epochs=[3, 9],
                   on_epoch=lambda m, e: snapshots.__setitem__(e, lm.logits_of(m, x)))
            short = lm.fit(x, y, OPT, 20, seed=4, epochs=[3])
            np.testing.assert_array_equal(snapshots[3], lm.logits_of(short, x))

    def test_mixture_starts_uniform_and_learns_layer_weights(self):
        x, y = separable(layers=3)
        x[:, 0] = torch.randn_like(x[:, 0])  # layer 0 is noise; layers 1-2 are informative
        model = lm.build_model(tuple(x.shape), 20, init_seed=0)
        self.assertTrue(torch.allclose(model.layer_weights(), torch.full((3,), 1 / 3)))
        trained = lm.fit(x, y, OPT, 20, seed=0, epochs=[200])
        weights = trained.layer_weights().detach()
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertLess(float(weights[0]), 1 / 3)

    def test_head_learns_separable_data(self):
        x, y = separable()
        model = lm.fit(x, y, OPT, 20, seed=0, epochs=[300])
        self.assertEqual(lm.classification_metrics(y.numpy(), lm.logits_of(model, x).argmax(1), 20)["accuracy"], 1.0)

    def test_macro_f1_uses_supported_labels_only(self):
        metrics = lm.classification_metrics([0, 0, 1, 1], [0, 0, 1, 2], 20)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (1.0 + 2 / 3) / 2)

    def test_exact_mcnemar(self):
        result = lm.paired([True] * 5 + [False] * 3, [False] * 5 + [False] * 3)
        self.assertEqual((result["only_arm_correct"], result["only_baseline_correct"]), (5, 0))
        self.assertAlmostEqual(result["exact_mcnemar_p"], 0.0625)
        self.assertEqual(lm.paired([True], [True])["exact_mcnemar_p"], 1.0)

    def test_selection_tie_breaks(self):
        specs = {"layer_0": {}, "layer_1": {}}
        rows = [{"arm": a, "epochs": e, "accuracy": 0.5, "macro_f1": 0.4} for a in specs for e in (2, 6)]
        summary, selected = lm.select(rows, specs, [2, 6])
        self.assertEqual(summary["layer_0"]["epochs"], 2)  # equal scores: fewer epochs
        self.assertEqual(selected, "layer_0")  # equal scores: lower layer index


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.train_meta, self.train_rows = write_split(self.root, "train", per_label=3)
        self.dev, self.dev_meta, self.dev_rows = write_split(self.root, "dev", per_label=1)
        self.config_path = self.root / "config.json"
        base.write_json(self.config_path, tiny_config())
        feature_config = clf.feature_config(tiny_config()["encoder"])
        torch.manual_seed(0)
        encoder = ae.AcousticEncoder(feature_config, TINY, device="cpu")
        self.cache = ae.FeatureCache(self.root / "cache", encoder.provenance())
        for rows in (self.train_rows, self.dev_rows):
            self.cache.write(encoder.extract_manifest(base.without_labels(rows), self.root / "audio"))

    def tearDown(self):
        self.temp.cleanup()

    def run_pilot(self, name, dev=None, dev_meta=None, output_root=None):
        argv = ["--train-manifest", str(self.train), "--train-metadata", str(self.train_meta),
                "--dev-manifest", str(dev or self.dev), "--dev-metadata", str(dev_meta or self.dev_meta),
                "--output-root", str(output_root or self.root / "cache"), "--labels", str(LABELS_PATH),
                "--config", str(self.config_path), "--output-dir", str(self.root / name)]
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = lm.main(argv, provenance_fn=tiny_provenance)
        return code, stderr.getvalue()

    def read(self, name, path):
        return (self.root / name / path).read_bytes()

    def test_end_to_end_outputs(self):
        code, err = self.run_pilot("run")
        self.assertEqual(code, 0, err)
        out = self.root / "run"
        for name in ("config.json", "folds.json", "cv_results.jsonl", "selection.json", "summary.json",
                     "layer_sweep.csv", "mixture_weights.json", "run_record.json"):
            self.assertTrue((out / name).is_file(), name)
        summary = json.loads((out / "summary.json").read_text())
        arms = [row["arm"] for row in summary["arms"]]
        self.assertEqual(arms, ["b0_final", "b0_final_raw", "selected_layer", "mixture_all", "mixture_middle"])
        for arm in arms:
            for seed in (1, 2):
                folder = out / "arms" / arm / f"seed_{seed}"
                predictions = ev.read_jsonl(folder / "predictions.jsonl")
                self.assertEqual([p["clip_id"] for p in predictions], [r["clip_id"] for r in self.dev_rows])
                self.assertTrue((folder / "evaluation/metrics.json").is_file())
        self.assertEqual(len((out / "layer_sweep.csv").read_text().splitlines()), 1 + 4)
        weights = json.loads((out / "mixture_weights.json").read_text())
        self.assertEqual(sorted(weights["mixture_middle"]["1"]), ["conformer_1", "conformer_2"])
        self.assertAlmostEqual(sum(weights["mixture_all"]["2"].values()), 1.0, places=5)
        record = json.loads((out / "run_record.json").read_text())
        self.assertFalse(record["features"]["encoder_loaded"])
        self.assertEqual(record["counts"]["cv_fits"], 7 * 2 * 3)  # (4 layers + raw + 2 mixtures) x seeds x folds

    def test_rerun_is_byte_identical(self):
        self.assertEqual(self.run_pilot("first")[0], 0)
        self.assertEqual(self.run_pilot("second")[0], 0)
        for path in ("selection.json", "cv_results.jsonl", "arms/mixture_all/seed_1/predictions.jsonl",
                     "arms/b0_final/seed_2/predictions.jsonl"):
            self.assertEqual(self.read("first", path), self.read("second", path), path)

    def test_dev_labels_never_change_choices_or_predictions(self):
        self.assertEqual(self.run_pilot("original")[0], 0)
        shuffled = [dict(r, label=self.dev_rows[(i + 7) % len(self.dev_rows)]["label"])
                    for i, r in enumerate(self.dev_rows)]
        manifest = self.root / "dev_shuffled.jsonl"
        base.write_jsonl(manifest, shuffled)
        metadata = json.loads(self.dev_meta.read_text())
        metadata["manifest_sha256"] = ev.sha256_file(manifest)
        meta_path = self.root / "dev_shuffled.metadata.json"
        base.write_json(meta_path, metadata)
        code, err = self.run_pilot("shuffled", dev=manifest, dev_meta=meta_path)
        self.assertEqual(code, 0, err)
        for path in ("selection.json", "arms/mixture_all/seed_1/predictions.jsonl",
                     "arms/selected_layer/seed_2/predictions.jsonl"):
            self.assertEqual(self.read("original", path), self.read("shuffled", path), path)

    def test_missing_cache_is_a_setup_error(self):
        code, err = self.run_pilot("run", output_root=self.root / "empty")
        self.assertEqual(code, 1)
        self.assertIn("No feature cache", err)
        self.assertFalse((self.root / "run").exists())

    def test_uncached_dev_clip_is_a_setup_error(self):
        extra = dict(self.dev_rows[0], clip_id="0" * 16)
        manifest = self.root / "dev_extra.jsonl"
        base.write_jsonl(manifest, self.dev_rows + [extra])
        metadata = json.loads(self.dev_meta.read_text())
        metadata["manifest_sha256"] = ev.sha256_file(manifest)
        meta_path = self.root / "dev_extra.metadata.json"
        base.write_json(meta_path, metadata)
        code, err = self.run_pilot("run", dev=manifest, dev_meta=meta_path)
        self.assertEqual(code, 1)
        self.assertIn("not cached", err)


if __name__ == "__main__":
    unittest.main()
