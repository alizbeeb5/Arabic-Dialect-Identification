"""Offline checks for fetching pilot WAVs from local stand-in Parquet shards."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import fetch_pilot_audio as fetch
from src.data import prepare

ROOT = Path(__file__).resolve().parents[1]


def fake_wav(n):
    return b"RIFF" + bytes([n % 251]) * (100 + n)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for shard_number, wanted in ((0, (1, 5, 9)), (3, (0, 4))):
            name = f"train-{shard_number:05d}-of-00038.parquet"
            path = self.root / "source/data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            audio = [{"bytes": fake_wav(shard_number * 100 + i), "path": None} for i in range(12)]
            # Small row groups so the reader must locate rows across several groups.
            pq.write_table(pa.table({"audio": audio, "dialect": ["MSA"] * 12}), path, row_group_size=4)
            for i in wanted:
                payload = audio[i]["bytes"]
                clip_id = prepare.make_clip_identity(name, i)
                self.rows.append({"clip_id": clip_id, "audio_path": f"train/{Path(name).stem}/{clip_id}.wav",
                                  "source_dataset": prepare.DATASET_ID,
                                  "source_revision": prepare.DATASET_REVISION,
                                  "source_shard": f"data/{name}", "source_row": i,
                                  "encoded_bytes": len(payload),
                                  "audio_sha256": hashlib.sha256(payload).hexdigest()})
        self.manifest = self.root / "pilot.jsonl"
        self.write_manifest(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run_fetch(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fetch.main(["--audio-root", str(self.root / "audio"), "--manifests", str(self.manifest),
                               "--source-root", str(self.root / "source")])

    def test_fetches_exact_bytes_and_resumes(self):
        self.assertEqual(self.run_fetch(), 0)
        for row in self.rows:
            data = (self.root / "audio" / row["audio_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["audio_sha256"])
        self.assertEqual(self.run_fetch(), 0)  # second run keeps verified files

    def test_hash_mismatch_writes_nothing_for_that_clip(self):
        self.rows[0]["audio_sha256"] = "0" * 64
        self.write_manifest(self.rows)
        self.assertEqual(self.run_fetch(), 1)
        self.assertFalse((self.root / "audio" / self.rows[0]["audio_path"]).exists())
        self.assertTrue((self.root / "audio" / self.rows[1]["audio_path"]).exists())

    def test_refuses_to_replace_a_differing_file(self):
        target = self.root / "audio" / self.rows[0]["audio_path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * self.rows[0]["encoded_bytes"])
        self.assertEqual(self.run_fetch(), 1)
        self.assertEqual(target.read_bytes(), b"x" * self.rows[0]["encoded_bytes"])

    def test_rejects_wrong_release(self):
        self.rows[0]["source_revision"] = "other"
        self.write_manifest(self.rows)
        self.assertEqual(self.run_fetch(), 1)

    def test_committed_pilot_rows_are_fetchable(self):
        rows = fetch.read_rows(fetch.DEFAULT_MANIFESTS)
        self.assertEqual(len(rows), 700)
        for row in rows:
            name = Path(row["source_shard"]).name
            self.assertEqual(prepare.make_clip_identity(name, row["source_row"]), row["clip_id"])


if __name__ == "__main__":
    unittest.main()
