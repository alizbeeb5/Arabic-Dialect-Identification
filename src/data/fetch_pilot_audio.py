#!/usr/bin/env python3
"""Fetch only the frozen pilot's WAV files from the pinned Hugging Face release.

prepare.py downloads all 44 source shards (~21.8 GB). The pilot needs 700 clips, and
every pilot manifest row records its source shard, source row and audio SHA256. This
script opens each needed shard remotely, reads only the Parquet row groups that
contain pilot rows (audio column only), and writes each clip's original WAV bytes to
the same relative path prepare.py uses. Every clip is checked against the manifest:
clip ID (recomputed from shard + row), byte length and audio SHA256. Nothing is
written for a clip that fails a check.

Already-present files with the expected length and SHA256 are kept, so the script can
be resumed. It refuses to replace a differing file.

Exit codes: 0 all clips present and verified, 1 setup error or any clip failed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402  (pinned dataset ID/revision, clip identity, atomic writes)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS = [ROOT / "data/manifests/train_pilot.jsonl", ROOT / "data/manifests/dev_pilot.jsonl"]


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    ids = [r["clip_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip_id across the given manifests")
    for row in rows:
        if (row.get("source_dataset") != prepare.DATASET_ID
                or row.get("source_revision") != prepare.DATASET_REVISION):
            raise ValueError(f"{row['clip_id']}: manifest row is not from the pinned release")
        relative = PurePosixPath(row["audio_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{row['clip_id']}: unsafe audio_path")
    return rows


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def present_and_valid(path, row):
    if not path.is_file():
        return False
    if path.stat().st_size != row["encoded_bytes"]:
        raise OSError(f"Existing file has a different size, not replacing it: {path}")
    if prepare.sha256_file(path) != row["audio_sha256"]:
        raise OSError(f"Existing file has a different SHA256, not replacing it: {path}")
    return True


def open_shard(source_root, shard):
    """source_root=None reads the pinned Hub revision; a local folder is for offline use/tests."""
    if source_root is not None:
        return open(Path(source_root) / shard, "rb")
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{prepare.DATASET_ID}@{prepare.DATASET_REVISION}/{shard}", "rb")


def fetch_shard(shard, wanted, audio_root, source_root):
    """wanted: {source_row: manifest row}. Returns (written, failures)."""
    import pyarrow.parquet as pq

    shard_name = PurePosixPath(shard).name
    written, failures = 0, []
    with open_shard(source_root, shard) as stream:
        parquet = pq.ParquetFile(stream)
        start = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            needed = sorted(r for r in wanted if start <= r < start + count)
            if needed:
                table = parquet.read_row_group(group, columns=["audio"])
                audio = table.column("audio")
                for source_row in needed:
                    row = wanted[source_row]
                    try:
                        if prepare.make_clip_identity(shard_name, source_row) != row["clip_id"]:
                            raise ValueError("clip_id does not match shard and row")
                        payload = audio[source_row - start].as_py()["bytes"]
                        if not payload or len(payload) != row["encoded_bytes"]:
                            raise ValueError("audio byte length differs from the manifest")
                        if sha256_bytes(payload) != row["audio_sha256"]:
                            raise ValueError("audio SHA256 differs from the manifest")
                        prepare.atomic_write(Path(audio_root) / row["audio_path"], payload)
                        written += 1
                    except (ValueError, KeyError, TypeError) as error:
                        failures.append({"clip_id": row["clip_id"], "shard": shard,
                                         "source_row": source_row, "error": str(error)})
            start += count
        missing_rows = [r for r in wanted if r >= start]
        failures += [{"clip_id": wanted[r]["clip_id"], "shard": shard, "source_row": r,
                      "error": "row index beyond the end of the shard"} for r in missing_rows]
    return written, failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Where WAVs go; pass this same folder as --audio-root to the experiments")
    parser.add_argument("--manifests", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Offline: local folder holding data/*.parquet instead of the Hub")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        rows = read_rows(args.manifests)
        todo = defaultdict(dict)
        kept = 0
        for row in rows:
            if present_and_valid(args.audio_root / row["audio_path"], row):
                kept += 1
            else:
                todo[row["source_shard"]][row["source_row"]] = row
    except (OSError, ValueError, KeyError) as error:
        print(f"Fetch error: {error}", file=sys.stderr)
        return 1

    prepare.progress(f"{len(rows)} pilot clips: {kept} already present and verified, "
                     f"{sum(map(len, todo.values()))} to fetch from {len(todo)} shards")
    written, failures = 0, []
    for number, shard in enumerate(sorted(todo), start=1):
        try:
            done, failed = fetch_shard(shard, todo[shard], args.audio_root, args.source_root)
        except Exception as error:  # noqa: BLE001 - report the shard and keep going
            done, failed = 0, [{"clip_id": r["clip_id"], "shard": shard, "source_row": i,
                                "error": f"{type(error).__name__}: {error}"} for i, r in todo[shard].items()]
        written += done
        failures += failed
        prepare.progress(f"[{number}/{len(todo)}] {shard}: wrote {done}, failed {len(failed)} "
                         f"({time.perf_counter() - started:.0f}s)")

    for failure in failures[:10]:
        print(f"FAILED {failure}", file=sys.stderr)
    prepare.progress(f"Done: {kept} kept, {written} written, {len(failures)} failed, "
                     f"{time.perf_counter() - started:.0f}s. Audio root: {args.audio_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
