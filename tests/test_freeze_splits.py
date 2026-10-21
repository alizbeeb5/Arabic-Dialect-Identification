"""Synthetic, offline checks for src/data/freeze_splits.py. No real data is read."""

import hashlib
from collections import Counter
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_splits", ROOT / "src/data/freeze_splits.py")
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)
LABELS = json.loads((ROOT / "configs/labels.json").read_text())["label_order"]
LABELS_HASH = hashlib.sha256((ROOT / "configs/labels.json").read_bytes()).hexdigest()


def clip(split, index, label, source=None, audio=None, peak=0.3):
    clip_id = hashlib.sha256(f"{split}-{index}".encode()).hexdigest()
    audio = audio or f"audio-{split}-{index}"
    return {"clip_id": clip_id, "audio_path": f"{split}/shard/{clip_id}.wav", "label": label,
            "split": split, "source_id": source, "speaker_id": None, "domain": "unknown",
            "duration_sec": 5.0, "sample_rate_hz": 16000, "audio_sha256": "a" + audio,
            "pcm_sha256": "p" + audio, "peak_abs": peak}


def write_release(folder, train, dev, pilot=False):
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("dev", dev)):
        path = folder / f"{split}.jsonl"
        path.write_bytes(freeze.jsonl_bytes(rows))
        meta = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "labels_sha256": LABELS_HASH, "dataset_version": "rev123",
                "split_version": "full-test", "audio_root": "/data/audio", "pilot": pilot,
                "source_provenance": {"dataset_id": "synthetic"}}
        path.with_suffix(".metadata.json").write_text(json.dumps(meta))


def standard_release():
    train, dev = [], []
    for i in range(20 * 40):  # 40 per label, 10 recordings per label in train
        label = LABELS[i % 20]
        train.append(clip("train", i, label, source=f"youtube:tr{label}{(i // 20) % 10}"))
    for i in range(20 * 15):
        label = LABELS[i % 20]
        dev.append(clip("dev", i, label, source=None if i % 3 else f"youtube:dv{i}"))
    # Leak 1: a dev clip from a training recording.
    dev.append(clip("dev", 900, "LEB", source="youtube:trLEB0"))
    # Leak 2: an exact audio duplicate across splits, no source_id.
    train.append(clip("train", 901, "EGY", audio="dup"))
    dev.append(clip("dev", 902, "EGY", audio="dup"))
    # A silent dev clip.
    dev.append(clip("dev", 903, "MOR", peak=0))
    return train, dev


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.release = self.root / "manifests"

    def tearDown(self):
        self.folder.cleanup()

    def run_freeze(self, *extra, out="out"):
        args = freeze.parse_args(["--manifest-dir", str(self.release),
                                  "--output-root", str(self.root / out), *extra])
        return freeze.run(args)

    def read(self, name, out="out"):
        path = self.root / out / "data/manifests" / name
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_pilots_are_balanced_disjoint_and_leak_free(self):
        write_release(self.release, *standard_release())
        self.run_freeze()
        train, dev = self.read("train_pilot.jsonl"), self.read("dev_pilot.jsonl")
        # 10 recordings per label: round-robin fills 25 per label at 2-3 clips per recording.
        self.assertEqual(len(train), 20 * 25)
        per_source = Counter(r["source_id"] for r in train)
        self.assertLessEqual(max(per_source.values()), 3)
        # LEB loses recording trLEB0 entirely because it leaks into dev; 9 remain, all used.
        self.assertNotIn("youtube:trLEB0", per_source)
        self.assertEqual(len({r["source_id"] for r in train if r["label"] == "LEB"}), 9)
        # Every pilot row is an unchanged row of its own upstream split.
        upstream = {r["clip_id"]: r for r in standard_release()[0]}
        self.assertTrue(all(upstream[r["clip_id"]] == r for r in train))
        self.assertEqual(len(dev), 200)
        self.assertEqual({r["label"] for r in dev}, set(LABELS))
        self.assertTrue(all(sum(r["label"] == l for r in dev) == 10 for l in LABELS))
        self.assertFalse({r["clip_id"] for r in train} & {r["clip_id"] for r in dev})
        train_sources = {r["source_id"] for r in train} - {None}
        self.assertFalse(train_sources & {r["source_id"] for r in dev})
        dev_ids = {r["clip_id"] for r in dev}
        for leaked in (clip("dev", 900, "LEB")["clip_id"], clip("dev", 902, "EGY")["clip_id"],
                       clip("dev", 903, "MOR")["clip_id"]):
            self.assertNotIn(leaked, dev_ids)
        self.assertNotIn(clip("train", 901, "EGY")["clip_id"], {r["clip_id"] for r in train})

    def test_reports_and_hash_record(self):
        write_release(self.release, *standard_release())
        record = self.run_freeze()
        out = self.root / "out"
        report = (out / "reports/split_audit.md").read_text()
        self.assertIn(record["split_version"], report)
        self.assertIn("speaker independence is not claimed", report)
        self.assertIn("not out-of-domain", report)
        crossing = [json.loads(l) for l in (out / "reports/split_cross_groups.jsonl").read_text().splitlines()]
        self.assertEqual(len(crossing), 2)
        reasons = {json.loads(l)["reason"] for l in (out / "reports/split_exclusions.jsonl").read_text().splitlines()}
        self.assertEqual(reasons, {"group_crosses_train_dev_boundary", "silent_audio"})
        sidecar = json.loads((out / "data/manifests/dev_pilot.metadata.json").read_text())
        self.assertEqual(sidecar["manifest_sha256"],
                         hashlib.sha256((out / "data/manifests/dev_pilot.jsonl").read_bytes()).hexdigest())
        self.assertEqual(sidecar["selection_split"], "dev_pilot")
        self.assertEqual(sorted(sidecar["evaluation_label_sets"]), ["dev"])  # evaluator looks up the row split
        self.assertEqual(sorted(record["dev_pilot_ids"]), sorted(r["clip_id"] for r in self.read("dev_pilot.jsonl")))
        self.assertIn("subset of upstream train", report)

    def test_deterministic_and_order_independent(self):
        train, dev = standard_release()
        write_release(self.release, train, dev)
        first = self.run_freeze(out="a")
        random.Random(7).shuffle(train)
        random.Random(8).shuffle(dev)
        write_release(self.release, train, dev)
        second = self.run_freeze(out="b")
        self.assertEqual(first["train_pilot_ids"], second["train_pilot_ids"])
        self.assertEqual(first["dev_pilot_ids"], second["dev_pilot_ids"])
        self.assertEqual(first["train_pilot_sha256"], second["train_pilot_sha256"])
        third = self.run_freeze("--seed", "7", out="c")
        self.assertNotEqual(first["dev_pilot_ids"], third["dev_pilot_ids"])

    def test_rejects_edited_manifest(self):
        write_release(self.release, *standard_release())
        with (self.release / "dev.jsonl").open("a") as stream:
            stream.write(json.dumps(clip("dev", 999, "LEB")) + "\n")
        with self.assertRaisesRegex(freeze.FreezeError, "SHA256"):
            self.run_freeze()

    def test_pilot_source_needs_explicit_flag(self):
        write_release(self.release, *standard_release(), pilot=True)
        with self.assertRaisesRegex(freeze.FreezeError, "bounded"):
            self.run_freeze()
        record = self.run_freeze("--allow-pilot-source")
        self.assertTrue(record["development_only"])
        self.assertIn("DEVELOPMENT-ONLY", (self.root / "out/reports/split_audit.md").read_text())

    def test_conflicting_labels_on_identical_audio_are_excluded(self):
        train, dev = standard_release()
        train.append(clip("train", 950, "KSA", audio="same"))
        train.append(clip("train", 951, "KUW", audio="same"))
        write_release(self.release, train, dev)
        self.run_freeze()
        reasons = [json.loads(l) for l in (self.root / "out/reports/split_exclusions.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["reason"] == "identical_audio_with_conflicting_labels" for r in reasons), 2)

    def test_hard_cap_creates_reported_shortfall(self):
        write_release(self.release, *standard_release())
        self.run_freeze("--max-per-group", "1")
        train = self.read("train_pilot.jsonl")
        self.assertEqual(len(train), 20 * 10 - 1)  # 10 recordings per label, LEB has 9
        report = (self.root / "out/reports/split_audit.md").read_text()
        self.assertIn("train LEB: wanted 25, selected 9", report)

    def test_few_recordings_still_fill_quota(self):
        train, dev = standard_release()
        dev = [r for r in dev if r["label"] != "BAH"]
        dev += [clip("dev", 1000 + i, "BAH", source=f"youtube:dvBAH{i % 4}") for i in range(40)]
        write_release(self.release, train, dev)
        self.run_freeze()
        bah = [r for r in self.read("dev_pilot.jsonl") if r["label"] == "BAH"]
        self.assertEqual(len(bah), 10)
        self.assertEqual(sorted(Counter(r["source_id"] for r in bah).values()), [2, 2, 3, 3])

    def test_rejects_duplicate_ids_and_bad_paths(self):
        train, dev = standard_release()
        dev.append(dict(train[0], split="dev"))
        write_release(self.release, train, dev)
        with self.assertRaisesRegex(freeze.FreezeError, "more than once"):
            self.run_freeze()
        train, dev = standard_release()
        train[0]["audio_path"] = "../escape.wav"
        write_release(self.release, train, dev)
        with self.assertRaisesRegex(freeze.FreezeError, "escapes"):
            self.run_freeze()


if __name__ == "__main__":
    unittest.main()
