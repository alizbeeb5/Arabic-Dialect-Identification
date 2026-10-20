#!/usr/bin/env python3
"""Download and audit the pinned ADI20 release on any writable filesystem.

CPU only. Run one preparation process, before launching training workers.
Existing audio is checked against source bytes; completed shard audits resume.
Upstream splits are retained. This does not construct the final research splits.
"""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

DATASET_ID = "UBC-NLP/NADI_2026_ADI20_micro"
DATASET_REVISION = "c8da454c9f474bb075d3fe608940ee29e8ccfa43"
EXPECTED_SHARDS = 44
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_RE = re.compile(r"data/(train|validation|dev|test)-(\d+)-of-(\d+)\.parquet")
SOURCE_RE = re.compile(
    r"(?P<id>[A-Za-z0-9_-]{11})_(?P<start>\d+(?:\.\d+)?)_"
    r"(?P<end>\d+(?:\.\d+)?)(?:_split_\d+)?\.wav"
)


class AudioAuditError(ValueError):
    """A source row cannot provide a usable labeled WAV."""


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepare-",
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_jsonl(path, rows):
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 allow_nan=False) + "\n" for row in rows)
    atomic_write(path, payload.encode("utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path,
                        default=REPO_ROOT / "configs" / "labels.json")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"],
                        help="Default: all splits actually present in the pinned release")
    parser.add_argument("--max-clips", type=int,
                        help="Maximum source rows per selected split (pilot only)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use existing shards and download_record.json, offline")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2,
                        help="Download threads; preparation itself is single process")
    args = parser.parse_args(argv)
    for name in ("batch_size", "workers", "max_clips"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.splits is not None and len(set(args.splits)) != len(args.splits):
        parser.error("--splits contains duplicates")
    return args


def load_labels(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = config["label_order"]
    if (not isinstance(labels, list) or len(labels) != 20
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != 20):
        raise ValueError("label_order must contain 20 unique strings")
    if config.get("num_labels") != 20 or config.get("index_base") != 0:
        raise ValueError("Expected num_labels=20 and index_base=0")
    if config["label_to_id"] != {label: i for i, label in enumerate(labels)}:
        raise ValueError("label_to_id disagrees with label_order")
    if config["id_to_label"] != {str(i): label for i, label in enumerate(labels)}:
        raise ValueError("id_to_label disagrees with label_order")
    return labels


def split_for_shard(name):
    match = SHARD_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected source shard path: {name}")
    return "dev" if match[1] == "validation" else match[1]


def validate_inventory(files, expected_count=EXPECTED_SHARDS):
    if len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} shards, found {len(files)}")
    names = [entry["path"] for entry in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate source shard paths")
    groups = defaultdict(list)
    for entry in files:
        split_for_shard(entry["path"])
        size = entry["bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("Missing or invalid shard byte size")
        match = SHARD_RE.fullmatch(entry["path"])
        groups[match[1]].append((int(match[2]), int(match[3])))
    if "validation" in groups and "dev" in groups:
        raise ValueError("Ambiguous validation and dev source shards")
    for group, indices in groups.items():
        totals = {total for _, total in indices}
        if len(totals) != 1 or sorted(i for i, _ in indices) != list(range(next(iter(totals)))):
            raise ValueError(f"Incomplete or inconsistent {group} shard sequence")
    return sorted(files, key=lambda entry: entry["path"])


def prepare_source(args, raw_root):
    """Use the old notebook's download record or download from the pinned Hub commit."""
    record_path = raw_root / "download_record.json"
    if args.skip_download:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("dataset_id") != DATASET_ID
                or record.get("dataset_revision") != DATASET_REVISION):
            raise ValueError("download_record.json identifies a different dataset/revision")
        files = validate_inventory(record["files"])
    else:
        # Cache environment is configured by main BEFORE this import.
        import huggingface_hub
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION,
                                    files_metadata=True, token=False)
        if info.sha != DATASET_REVISION:
            raise ValueError("Hub returned a different revision")
        files = []
        for sibling in info.siblings:
            if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet"):
                entry = {"path": sibling.rfilename, "bytes": sibling.size}
                lfs_hash = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
                if lfs_hash:
                    entry["expected_sha256"] = lfs_hash
                files.append(entry)
        files = validate_inventory(files)
        snapshot_download(repo_id=DATASET_ID, repo_type="dataset",
                          revision=DATASET_REVISION, local_dir=str(raw_root),
                          allow_patterns=[entry["path"] for entry in files] + ["README.md"],
                          max_workers=args.workers, token=False)
        record = {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                  "huggingface_hub_version": huggingface_hub.__version__, "files": files,
                  "shard_count": len(files), "finished_at_utc": now()}
    for entry in files:
        path = raw_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size shard: {entry['path']}")
    if not args.skip_download:
        record.update(file_sizes_verified=True, content_hashes_verified=False,
                      audio_audit_completed=False,
                      total_bytes=sum(entry["bytes"] for entry in files))
        atomic_write(record_path, json_bytes(record))
    return files


def make_clip_identity(shard_name, row_index):
    key = f"{DATASET_ID}\n{DATASET_REVISION}\ndata/{shard_name}\n{row_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def inspect_audio_bytes(payload, block_frames=65536):
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise AudioAuditError("Missing embedded audio bytes")
    if type(block_frames) is not int or block_frames <= 0:
        raise ValueError("block_frames must be a positive integer")
    pcm_hash = hashlib.sha256()
    frames = 0
    peak = square_sum = clipped_samples = 0.0
    try:
        with sf.SoundFile(io.BytesIO(payload)) as stream:
            rate, channels, expected = stream.samplerate, stream.channels, stream.frames
            audio_format, subtype = stream.format, stream.subtype
            if rate <= 0 or channels <= 0:
                raise AudioAuditError("Invalid sample rate or channels")
            if audio_format != "WAV":
                raise AudioAuditError(f"Expected WAV, found {audio_format}")
            pcm_hash.update(f"pcm-f64le-v1:{rate}:{channels}:".encode("ascii"))
            for block in stream.blocks(blocksize=block_frames, dtype="float64", always_2d=True):
                if not np.isfinite(block).all():
                    raise AudioAuditError("Nonfinite samples")
                frames += len(block)
                pcm_hash.update(block.astype("<f8", copy=False).tobytes(order="C"))
                peak = max(peak, float(np.abs(block).max(initial=0)))
                square_sum += float(np.sum(block * block))
                clipped_samples += int(np.count_nonzero(np.abs(block) >= 0.999))
    except (sf.LibsndfileError, RuntimeError, ValueError) as error:
        raise AudioAuditError(str(error)) from error
    if frames == 0 or frames != expected:
        raise AudioAuditError("Empty audio or decoded frame count differs from header")
    return {"duration_sec": frames / rate, "sample_rate_hz": rate,
            "channel_count": channels, "frame_count": frames, "audio_format": audio_format,
            "audio_subtype": subtype, "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "pcm_sha256": pcm_hash.hexdigest(), "peak_abs": peak,
            "rms": math.sqrt(square_sum / (frames * channels)),
            "near_full_scale_fraction": clipped_samples / (frames * channels),
            "encoded_bytes": len(payload)}


def source_metadata(original_path):
    name = PurePosixPath((original_path or "").replace("\\", "/")).name
    match = SOURCE_RE.fullmatch(name)
    result = {"source_id": None, "source_id_method": "unknown",
              "speaker_id": None, "domain": "unknown"}
    if match and float(match["end"]) > float(match["start"]):
        result.update(source_id=f"youtube:{match['id']}",
                      source_id_method="inferred_from_filename_not_verified")
    return result


def extract_clip(row, split, shard_name, row_index, audio_root, labels):
    label = row.get("dialect")
    if not isinstance(label, str) or label not in labels:
        raise AudioAuditError(f"Unknown or missing dialect label: {label!r}")
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise AudioAuditError("Missing audio record")
    original = audio.get("path")
    if original is not None and not isinstance(original, str):
        raise AudioAuditError("Original audio path is not a string")
    payload = audio.get("bytes")
    inspection = inspect_audio_bytes(payload)
    clip_id = make_clip_identity(shard_name, row_index)
    relative = str(PurePosixPath(split) / Path(shard_name).stem / f"{clip_id}.wav")
    destination = audio_root / relative
    if destination.exists():
        if sha256_file(destination) != inspection["audio_sha256"]:
            raise OSError(f"Existing extracted audio differs: {destination}; inspect before replacing it")
    else:
        atomic_write(destination, payload)
        if sha256_file(destination) != inspection["audio_sha256"]:
            raise OSError(f"Written audio failed hash verification: {destination}")
    return {"clip_id": clip_id, "audio_path": relative, "label": label, "split": split,
            "source_dataset": DATASET_ID, "source_revision": DATASET_REVISION,
            "source_shard": f"data/{shard_name}", "source_row": row_index,
            "original_path": original, **source_metadata(original), **inspection}


def audit_shard(path, source_hash, split, limit, labels, audio_root, state_root, batch_size):
    cache_path = state_root / f"{path.stem}-limit-{limit or 'all'}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["source_sha256"] != source_hash:
            raise ValueError(f"Source changed since audit: {path.name}")
        if all((audio_root / row["audio_path"]).is_file()
               and sha256_file(audio_root / row["audio_path"]) == row["audio_sha256"]
               for row in cached["records"]):
            print(f"Reused audited shard: {path.name}", flush=True)
            return cached
        print(f"Rechecking shard with missing/changed extracted files: {path.name}", flush=True)
    records, exclusions = [], []
    with pq.ParquetFile(path) as parquet:
        if not {"audio", "dialect"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing audio/dialect columns: {path.name}")
        total_rows = parquet.metadata.num_rows
        count = total_rows if limit is None else min(total_rows, limit)
        row_index = 0
        if count:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["audio", "dialect"]):
                for row in batch.to_pylist():
                    try:
                        records.append(extract_clip(row, split, path.name, row_index, audio_root, labels))
                    except AudioAuditError as error:
                        exclusions.append({"clip_id": make_clip_identity(path.name, row_index),
                                           "source_shard": f"data/{path.name}",
                                           "source_row": row_index, "split": split,
                                           "label": row.get("dialect"),
                                           "reason": str(error), "status": "excluded"})
                    row_index += 1
                    if row_index == count:
                        break
                if row_index == count:
                    break
        if row_index != count:
            raise ValueError(f"Parquet row count mismatch: {path.name}")
    result = {"source_sha256": source_hash, "source_rows": total_rows,
              "attempted_rows": count, "records": records, "exclusions": exclusions}
    atomic_write(cache_path, json_bytes(result))
    print(f"Audited {path.name}: {len(records)} accepted, {len(exclusions)} excluded", flush=True)
    return result


def find_overlaps(records):
    findings = []
    for kind, field in [("encoded_exact", "audio_sha256"), ("decoded_exact", "pcm_sha256"),
                        ("inferred_source", "source_id")]:
        groups = defaultdict(list)
        for row in records:
            if row[field] is not None:
                groups[row[field]].append({"clip_id": row["clip_id"], "split": row["split"],
                                           "label": row["label"]})
        for value, members in groups.items():
            if len(members) > 1:
                findings.append({"kind": kind, "value": value, "members": members,
                                 "cross_split": len({m["split"] for m in members}) > 1,
                                 "label_conflict": len({m["label"] for m in members}) > 1})
    return findings


def summarize(records, labels):
    durations = [row["duration_sec"] for row in records]
    return {"clips": len(records), "label_counts": {label: sum(r["label"] == label for r in records)
                                                    for label in labels},
            "duration_hours": sum(durations) / 3600,
            "duration_min_sec": min(durations, default=None),
            "duration_max_sec": max(durations, default=None),
            "sample_rate_counts": dict(Counter(str(r["sample_rate_hz"]) for r in records)),
            "channel_counts": dict(Counter(str(r["channel_count"]) for r in records)),
            "subtype_counts": dict(Counter(r["audio_subtype"] for r in records)),
            "silent_clips": sum(r["peak_abs"] == 0 for r in records),
            "clips_with_near_full_scale_samples": sum(r["near_full_scale_fraction"] > 0 for r in records),
            "unknown_source_ids": sum(r["source_id"] is None for r in records),
            "unknown_speaker_ids": sum(r["speaker_id"] is None for r in records),
            "extracted_bytes": sum(r["encoded_bytes"] for r in records)}


def run(args):
    started_at = now()
    labels = load_labels(args.labels)
    root = args.data_root.expanduser().resolve()
    raw_root = root / "raw" / "adi20_micro" / DATASET_REVISION
    audio_root = root / "audio" / "adi20_micro" / DATASET_REVISION
    files = prepare_source(args, raw_root)
    available = {split_for_shard(f["path"]) for f in files}
    if args.splits is None:
        args.splits = [s for s in ("train", "dev", "test") if s in available]
    missing = set(args.splits) - available
    if missing:
        raise ValueError(f"Requested splits unavailable in pinned release: {sorted(missing)}; "
                         f"available: {sorted(available)}. Pass --splits explicitly.")
    code_hash, labels_hash = sha256_file(__file__), sha256_file(args.labels)
    identity = {"schema_version": "1.0", "dataset_revision": DATASET_REVISION,
                "code_sha256": code_hash, "labels_sha256": labels_hash,
                "splits": sorted(args.splits), "max_clips_per_split": args.max_clips,
                "inventory": files}
    release_id = ("pilot" if args.max_clips else "full") + "-" + hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    manifest_root = root / "manifests" / "adi20_micro" / DATASET_REVISION / release_id
    report_root = root / "reports" / "adi20_micro" / DATASET_REVISION / release_id
    state_root = report_root / "shard_state"
    report_root.mkdir(parents=True, exist_ok=True)
    lock = report_root / ".prepare.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Preparation lock exists: {lock}. If a previous process was interrupted, "
                           "confirm it has stopped, remove ONLY that empty lock directory, and rerun.") from error
    try:
        atomic_write(report_root / "status.json", json_bytes({"status": "running", "started_at": started_at}))
        records, exclusions, inventory = [], [], []
        attempted = Counter()
        total_rows = Counter()
        for entry in files:
            split = split_for_shard(entry["path"])
            if split not in args.splits:
                continue
            path = raw_root / entry["path"]
            # Always read Parquet metadata, even when a pilot has reached its limit.
            with pq.ParquetFile(path) as parquet:
                total_rows[split] += parquet.metadata.num_rows
            remaining = None if args.max_clips is None else args.max_clips - attempted[split]
            if remaining is not None and remaining <= 0:
                continue
            source_hash = sha256_file(path)
            if entry.get("expected_sha256") and source_hash != entry["expected_sha256"]:
                raise ValueError(f"Source content differs from Hub LFS hash: {entry['path']}")
            result = audit_shard(path, source_hash, split, remaining, labels,
                                 audio_root, state_root, args.batch_size)
            records.extend(result["records"])
            exclusions.extend(result["exclusions"])
            attempted[split] += result["attempted_rows"]
            inventory.append({**entry, "sha256": source_hash,
                              "hub_hash_verified": bool(entry.get("expected_sha256"))})
        if len({r["clip_id"] for r in records}) != len(records):
            raise ValueError("Duplicate clip IDs")
        overlaps = find_overlaps(records)
        write_jsonl(report_root / "overlaps.jsonl", overlaps)
        write_jsonl(report_root / "exclusions.jsonl", exclusions)
        support = {split: [label for label in labels if any(r["label"] == label and r["split"] == split
                                                           for r in records)] for split in args.splits}
        manifests = {}
        for split in args.splits:
            selected = [r for r in records if r["split"] == split]
            path = manifest_root / f"{split}.jsonl"
            write_jsonl(path, selected)
            metadata = {"schema_version": "1.0", "dataset_version": DATASET_REVISION,
                        "split_version": release_id, "manifest_sha256": sha256_file(path),
                        "labels_sha256": labels_hash, "audio_root": str(audio_root),
                        "audio_root_relative_to_data_root": f"audio/adi20_micro/{DATASET_REVISION}",
                        "evaluation_label_sets": support, "selection_split": None,
                        "split": split, "pilot": args.max_clips is not None,
                        "attempted_rows": attempted[split], "accepted_rows": len(selected),
                        "excluded_rows": sum(e["split"] == split for e in exclusions),
                        "source_provenance": {"dataset_id": DATASET_ID, "revision": DATASET_REVISION},
                        "source_independence_verified": False, "speaker_independence_verified": False,
                        "split_policy": "upstream_splits_retained_pending_Task_7",
                        "preparation_code_sha256": code_hash,
                        "extensions": {"source_dataset": "string", "source_revision": "string",
                                       "source_shard": "string", "source_row": "integer",
                                       "original_path": "string|null", "source_id_method": "string",
                                       "channel_count": "integer", "frame_count": "integer",
                                       "audio_format": "string", "audio_subtype": "string",
                                       "audio_sha256": "string", "pcm_sha256": "string",
                                       "peak_abs": "number", "rms": "number",
                                       "near_full_scale_fraction": "number", "encoded_bytes": "integer"}}
            atomic_write(path.with_suffix(".metadata.json"), json_bytes(metadata))
            manifests[split] = {"path": str(path), "sha256": metadata["manifest_sha256"]}
        audit = {"status": "completed_with_exclusions" if exclusions else "completed",
                 "release_id": release_id, "started_at_utc": started_at, "finished_at_utc": now(),
                 "pilot": args.max_clips is not None, "source_rows_by_split": dict(total_rows),
                 "attempted_rows_by_split": dict(attempted), "exclusions": len(exclusions),
                 "summary": summarize(records, labels),
                 "per_split": {s: summarize([r for r in records if r["split"] == s], labels) for s in args.splits},
                 "overlap_groups": dict(Counter(o["kind"] for o in overlaps)),
                 "cross_split_overlap_groups": dict(Counter(o["kind"] for o in overlaps if o["cross_split"])),
                 "near_duplicate_audit_completed": False, "source_ids_verified": False,
                 "speaker_independence_verified": False, "manifests": manifests,
                 "audio_root": str(audio_root), "processed_shards": inventory,
                 "download_bytes": sum(e["bytes"] for e in files),
                 "versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pyarrow": pa.__version__, "soundfile": sf.__version__,
                              "libsndfile": sf.__libsndfile_version__},
                 "code_sha256": code_hash, "labels_sha256": labels_hash,
                 "review_required_before_training": True,
                 "notes": ["Original WAV bytes preserved; no resampling, trimming or normalization.",
                           "Invalid rows are explicit exclusions; do not silently alter a fixed evaluation denominator.",
                           "Filename source IDs are inferences; speakers and domain remain unknown.",
                           "Exact duplicates are reported, not removed. Near-duplicates and frozen splits belong to Task 7.",
                           "A pilot takes the first rows; it is not a balanced or representative dataset.",
                           "Source SHA256 values are local fingerprints unless hub_hash_verified is true."]}
        atomic_write(report_root / "audit.json", json_bytes(audit))
        atomic_write(report_root / "status.json", json_bytes({"status": audit["status"], "finished_at": now()}))
        print(f"\n{audit['status']}: {len(records)} clips, {len(exclusions)} exclusions", flush=True)
        print(f"MANIFESTS: {manifest_root}\nAUDIT: {report_root / 'audit.json'}", flush=True)
        return audit
    except BaseException:
        atomic_write(report_root / "status.json", json_bytes({"status": "interrupted_or_failed", "at": now()}))
        raise
    finally:
        lock.rmdir()


def main():
    args = parse_args()
    # Must precede importing huggingface_hub in this fresh Python process.
    cache = (args.cache_root or args.data_root / "cache").expanduser().resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    audit = run(args)
    return 2 if audit["exclusions"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
