"""Synthetic, offline checks. No dataset or GPU is downloaded."""

import contextlib
import importlib.util
import io
import json
import hashlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/data/prepare.py"
SPEC = importlib.util.spec_from_file_location("prepare", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
LABELS = json.loads((ROOT / "configs/labels.json").read_text())["label_order"]


def wav_payload(offset=0):
    output = io.BytesIO()
    samples = 0.2 * np.sin(np.arange(3200) * 0.13 + offset)
    sf.write(output, samples, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.payload = wav_payload()

    def tearDown(self):
        self.folder.cleanup()

    def row(self, label="LEB", payload=None):
        return {"dialect": label, "audio": {"bytes": self.payload if payload is None else payload,
                                             "path": "nO0zLzlSLvs_1086.50_1092.19.wav"}}

    def extract(self, row=None, index=0):
        return prepare.extract_clip(row or self.row(), "train", "train-00000-of-00038.parquet",
                                    index, self.root / "audio", LABELS)

    def make_release(self, invalid=False):
        raw = self.root / "raw/adi20_micro" / prepare.DATASET_REVISION
        (raw / "data").mkdir(parents=True)
        files = []
        for split, total in [("train", 38), ("test", 6)]:
            for index in range(total):
                name = f"data/{split}-{index:05d}-of-{total:05d}.parquet"
                rows = [self.row(), self.row("EGY", wav_payload(index + 1))]
                if invalid and split == "train" and index == 0:
                    rows[0] = self.row(payload=b"invalid audio")
                pq.write_table(pa.Table.from_pylist(rows), raw / name)
                files.append({"path": name, "bytes": (raw / name).stat().st_size})
        record = {"dataset_id": prepare.DATASET_ID, "dataset_revision": prepare.DATASET_REVISION,
                  "files": files, "shard_count": 44, "file_sizes_verified": True}
        (raw / "download_record.json").write_text(json.dumps(record))
        return raw, files

    def args(self, *extra):
        return prepare.parse_args(["--data-root", str(self.root), "--skip-download", *extra])

    def run_quietly(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare.run(self.args(*extra))

    def test_label_contract_and_mismatched_mapping(self):
        self.assertEqual(prepare.load_labels(ROOT / "configs/labels.json"), LABELS)
        config = json.loads((ROOT / "configs/labels.json").read_text())
        config["label_to_id"]["LEB"] = 0
        path = self.root / "bad-labels.json"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "label_to_id"):
            prepare.load_labels(path)

    def test_hashes_and_audio_preservation(self):
        row = self.extract()
        self.assertEqual((self.root / "audio" / row["audio_path"]).read_bytes(), self.payload)
        small = prepare.inspect_audio_bytes(self.payload, block_frames=17)
        self.assertEqual(small["pcm_sha256"], row["pcm_sha256"])
        self.assertEqual(row["sample_rate_hz"], 16000)
        self.assertAlmostEqual(row["duration_sec"], 0.2)
        self.assertFalse(Path(row["audio_path"]).is_absolute())
        self.assertEqual(row["source_id"], "youtube:nO0zLzlSLvs")
        self.assertIsNone(row["speaker_id"])

    def test_repeated_extraction_and_conflicting_output(self):
        first = self.extract()
        self.assertEqual(first, self.extract())
        (self.root / "audio" / first["audio_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(OSError, "differs"):
            self.extract()

    def test_source_row_identity_and_untrusted_original_name(self):
        self.assertNotEqual(self.extract()["clip_id"], self.extract(index=1)["clip_id"])
        row = self.row()
        row["audio"]["path"] = "../../outside.wav"
        result = self.extract(row)
        self.assertNotIn("..", Path(result["audio_path"]).parts)
        self.assertIsNone(result["source_id"])

    def test_invalid_labels_and_corrupt_audio_rejected(self):
        with self.assertRaises(prepare.AudioAuditError):
            self.extract(self.row("INVALID"))
        with self.assertRaises(prepare.AudioAuditError):
            self.extract(self.row(payload=b"broken"))

    def test_pilot_full_resume_and_portable_manifests(self):
        self.make_release()
        pilot = self.run_quietly("--splits", "train", "--max-clips", "1")
        self.assertTrue(pilot["pilot"])
        self.assertEqual(pilot["summary"]["clips"], 1)
        first = self.run_quietly()
        self.assertFalse(first["pilot"])
        self.assertEqual(first["summary"]["clips"], 88)
        self.assertEqual(first["attempted_rows_by_split"], {"train": 76, "test": 12})
        second = self.run_quietly()
        self.assertEqual(first["manifests"], second["manifests"])
        self.assertTrue(first["cross_split_overlap_groups"]["decoded_exact"] > 0)
        path = Path(first["manifests"]["train"]["path"])
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["manifest_sha256"], prepare.sha256_file(path))
        self.assertEqual(metadata["evaluation_label_sets"]["train"], ["EGY", "LEB"])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            self.assertTrue((Path(metadata["audio_root"]) / row["audio_path"]).is_file())

    def test_exclusions_are_explicit_and_accounted_for(self):
        self.make_release(invalid=True)
        audit = self.run_quietly("--splits", "train", "--max-clips", "2")
        self.assertEqual(audit["status"], "completed_with_exclusions")
        self.assertEqual(audit["exclusions"], 1)
        self.assertEqual(audit["summary"]["clips"], 1)
        self.assertEqual(audit["attempted_rows_by_split"]["train"], 2)
        reports = self.root / "reports/adi20_micro" / prepare.DATASET_REVISION / audit["release_id"]
        exclusion = json.loads((reports / "exclusions.jsonl").read_text())
        self.assertEqual(exclusion["source_row"], 0)

    def test_missing_audio_is_recreated_from_source_on_resume(self):
        self.make_release()
        audit = self.run_quietly("--splits", "train", "--max-clips", "1")
        row = json.loads(Path(audit["manifests"]["train"]["path"]).read_text())
        audio = Path(audit["audio_root"]) / row["audio_path"]
        audio.unlink()
        self.run_quietly("--splits", "train", "--max-clips", "1")
        self.assertEqual(audio.read_bytes(), self.payload)

    def test_mismatched_revision_and_missing_shard_stop(self):
        raw, _ = self.make_release()
        path = raw / "download_record.json"
        record = json.loads(path.read_text())
        record["dataset_revision"] = "wrong"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "different dataset"):
            self.run_quietly()
        record["dataset_revision"] = prepare.DATASET_REVISION
        path.write_text(json.dumps(record))
        (raw / record["files"][0]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing or wrong-size"):
            self.run_quietly()

    def test_inventory_and_unknown_split_are_not_silently_accepted(self):
        _, files = self.make_release()
        with self.assertRaisesRegex(ValueError, "Expected 44"):
            prepare.validate_inventory(files[:-1])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.run_quietly("--splits", "dev")

    def test_cli_runs_offline_and_returns_audit_exit_code(self):
        self.make_release(invalid=True)
        command = [sys.executable, str(SCRIPT), "--data-root", str(self.root),
                   "--skip-download", "--splits", "train", "--max-clips", "2"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("completed_with_exclusions", result.stdout)

    def test_online_download_interface_uses_pinned_revision(self):
        raw, files = self.make_release()
        info = SimpleNamespace(sha=prepare.DATASET_REVISION, siblings=[
            SimpleNamespace(rfilename=f["path"], size=f["bytes"], lfs=None) for f in files])
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as download:
            api.return_value.dataset_info.return_value = info
            args = self.args()
            args.skip_download = False
            self.assertEqual(prepare.prepare_source(args, raw), sorted(files, key=lambda f: f["path"]))
            self.assertEqual(download.call_args.kwargs["revision"], prepare.DATASET_REVISION)
            self.assertEqual(download.call_args.kwargs["local_dir"], str(raw))

    def test_size_mode_is_explicit_and_skips_wav_content_reads(self):
        self.make_release()
        self.run_quietly("--splits", "train", "--max-clips", "1")
        hashed_wavs = []
        real_hash = prepare.sha256_file
        def counted(path):
            if Path(path).suffix == ".wav":
                hashed_wavs.append(str(path))
            return real_hash(path)
        with patch.object(prepare, "sha256_file", side_effect=counted):
            audit = self.run_quietly("--splits", "train", "--max-clips", "1", "--verification", "size")
        self.assertEqual(hashed_wavs, [])
        self.assertFalse(audit["stored_audio_content_hashes_verified_this_run"])
        self.assertEqual(audit["stored_audio_verification"], "size")

    def test_same_size_corruption_is_detected_by_full_mode(self):
        row = self.extract()
        path = self.root / "audio" / row["audio_path"]
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        self.assertTrue(prepare.verify_cached_audio([row], self.root / "audio", "size"))
        self.assertFalse(prepare.verify_cached_audio([row], self.root / "audio", "sha256"))

    def test_local_staging_preserves_hash_and_cleans_up(self):
        self.make_release()
        audit = self.run_quietly("--splits", "train", "--max-clips", "1",
                                 "--work-root", str(self.root / "temporary"))
        self.assertEqual(audit["summary"]["clips"], 1)
        self.assertEqual(list((self.root / "temporary").rglob("*.parquet")), [])
        source = self.root / "source.parquet"
        source.write_bytes(b"synthetic parquet bytes")
        local, digest = prepare.stage_shard(source, self.root / "stage")
        self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(local.read_bytes(), source.read_bytes())
        with self.assertRaisesRegex(ValueError, "must differ"):
            prepare.stage_shard(source, source.parent)

    def test_interrupted_v1_fourteen_shards_migrate_without_changing_manifest_rows(self):
        self.make_release()
        legacy_path = ROOT / "tests/fixtures/prepare_v1.py"
        spec = importlib.util.spec_from_file_location("legacy_prepare", legacy_path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        self.assertEqual(prepare.LEGACY_CODE_SHA256, legacy.sha256_file(legacy_path))
        old_args = legacy.parse_args(["--data-root", str(self.root), "--skip-download",
                                      "--labels", str(ROOT / "configs/labels.json")])
        completed = 0
        original_audit = legacy.audit_shard
        def interrupted(*args, **kwargs):
            nonlocal completed
            if completed == 14:
                raise KeyboardInterrupt("synthetic interruption")
            result = original_audit(*args, **kwargs)
            completed += 1
            return result
        with patch.object(legacy, "audit_shard", side_effect=interrupted):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                legacy.run(old_args)
        old_report = next((self.root / "reports/adi20_micro" / prepare.DATASET_REVISION).glob("full-*"))
        old_states = list((old_report / "shard_state").glob("*.json"))
        self.assertEqual(len(old_states), 14)
        original_records = {r["clip_id"]: r for p in old_states
                            for r in json.loads(p.read_text())["records"]}
        audit = self.run_quietly("--resume-from", str(old_report), "--verification", "size",
                                 "--work-root", str(self.root / "temporary"))
        self.assertEqual(audit["imported_shard_audits"], 14)
        self.assertEqual(audit["summary"]["clips"], 88)
        migrated = {r["clip_id"]: r for m in audit["manifests"].values()
                    for r in map(json.loads, Path(m["path"]).read_text().splitlines())}
        for clip_id, row in original_records.items():
            self.assertEqual(migrated[clip_id], row)
        self.assertEqual(len(list((old_report / "shard_state").glob("*.json"))), 14)
        verified = self.run_quietly("--resume-from", str(old_report), "--verification", "sha256")
        self.assertEqual(verified["manifests"], audit["manifests"])
        self.assertTrue(verified["stored_audio_content_hashes_verified_this_run"])

    def test_incompatible_legacy_directory_is_rejected(self):
        self.make_release()
        with self.assertRaisesRegex(ValueError, "Incompatible --resume-from"):
            self.run_quietly("--resume-from", str(self.root / "unrelated"))

    def test_failure_status_records_the_actual_error_and_phase(self):
        self.make_release()
        with patch.object(prepare, "audit_shard", side_effect=OSError("synthetic storage failure")):
            with self.assertRaisesRegex(OSError, "synthetic storage"):
                self.run_quietly()
        status_path = next((self.root / "reports").rglob("status.json"))
        status = json.loads(status_path.read_text())
        self.assertEqual(status["error_type"], "OSError")
        self.assertEqual(status["error_message"], "synthetic storage failure")
        self.assertIn("last_phase", status)
        self.assertFalse((status_path.parent / ".prepare.lock").exists())

    def test_stop_helper_targets_only_matching_dataset_and_preserves_data(self):
        helper = ROOT / "src/data/stop_prepare.py"
        dummy = self.root / "dummy" / "prepare.py"
        dummy.parent.mkdir()
        marker = self.root / "ready"
        dummy.write_text("import time\nfrom pathlib import Path\n"
                         f"Path({str(marker)!r}).touch()\n"
                         "try:\n while True: time.sleep(0.1)\n"
                         "except KeyboardInterrupt: pass\n")
        child = subprocess.Popen([sys.executable, str(dummy), "--data-root", str(self.root)])
        try:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            lock = self.root / "reports/adi20_micro/test/.prepare.lock"
            lock.mkdir(parents=True)
            result = subprocess.run([sys.executable, str(helper), "--data-root", str(self.root),
                                     "--timeout", "3", "--clear-lock", str(lock)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            child.wait(timeout=3)
            self.assertFalse(lock.exists())
            self.assertTrue(marker.exists())
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
