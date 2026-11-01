#!/usr/bin/env python3
"""Download, audit and freeze the pinned Casablanca validation split.

CPU only. Reads the 15 validation shards; Casablanca test is never downloaded or read.
Original WAV bytes are extracted unchanged. Rules: docs/casablanca_validation.md.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare as adi  # noqa: E402  (PCM fingerprint, atomic writes, label checks shared with ADI-20)

DATASET_ID = "UBC-NLP/Casablanca"
DATASET_REVISION = "8951b1b88e28c1107142ced57967b8d16350951d"
PIPELINE_VERSION = "casablanca-validation-audit-v1"  # Bump when identity, decoding or manifest fields change.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_TO_LABEL = {"Algeria": "ALG", "Egypt": "EGY", "Jordan": "JOR", "Mauritania": "MAU",
                   "Morocco": "MOR", "Palestine": "PAL", "UAE": "UAE", "Yemen": "YEM"}
PUBLISHED_VALIDATION_COUNTS = {"Algeria": 844, "Egypt": 846, "Jordan": 848, "Mauritania": 953,
                               "Morocco": 1045, "Palestine": 667, "UAE": 813, "Yemen": 803}
EXPECTED_SHARDS = 15
SHARD_RE = re.compile(r"(?P<config>[A-Za-z]+)/validation-(?P<index>\d+)-of-(?P<total>\d+)\.parquet")
SEG_ID_RE = re.compile(r"(?P<prefix>\d+)_(?P<index>\d+)")
DURATION_TOLERANCE_SEC = 0.010
DOMAIN = "tv_drama"
ADI20_MANIFESTS = [REPO_ROOT / "src/data/manifests/train.jsonl", REPO_ROOT / "src/data/manifests/dev.jsonl"]
RULES_DOC = REPO_ROOT / "docs/casablanca_validation.md"
COLUMNS = ["audio", "seg_id", "transcription", "gender", "duration"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Writable storage for raw shards and extracted audio (outside Git)")
    parser.add_argument("--raw-dir", type=Path,
                        help="Existing copy of the pinned release. Default: DATA_ROOT/raw/casablanca/<revision>")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use shards already in the raw directory; only Hub file metadata is fetched")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    parser.add_argument("--adi20-manifests", type=Path, nargs="+", default=ADI20_MANIFESTS)
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifests")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports" / "casablanca_validation_audit")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="Download threads")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers <= 0:
        parser.error("--batch-size and --workers must be positive")
    return args


def validate_inventory(files):
    """Exactly the 15 validation shards of the eight configurations; anything test-like is refused."""
    if any("test" in entry["path"].lower() for entry in files):
        raise ValueError("Test files must not enter Casablanca validation preparation")
    if len(files) != EXPECTED_SHARDS:
        raise ValueError(f"Expected {EXPECTED_SHARDS} validation shards, found {len(files)}")
    groups = defaultdict(list)
    for entry in files:
        match = SHARD_RE.fullmatch(entry["path"])
        if not match or match["config"] not in CONFIG_TO_LABEL:
            raise ValueError(f"Unexpected validation shard path: {entry['path']}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Missing or invalid shard byte size: {entry['path']}")
        groups[match["config"]].append((int(match["index"]), int(match["total"])))
    if set(groups) != set(CONFIG_TO_LABEL):
        raise ValueError(f"Missing configurations: {sorted(set(CONFIG_TO_LABEL) - set(groups))}")
    for config, indices in groups.items():
        totals = {total for _, total in indices}
        if len(totals) != 1 or sorted(i for i, _ in indices) != list(range(next(iter(totals)))):
            raise ValueError(f"Incomplete or inconsistent {config} validation shard sequence")
    if len({entry["path"] for entry in files}) != len(files):
        raise ValueError("Duplicate validation shard paths")
    return sorted(files, key=lambda entry: entry["path"])


def hub_inventory():
    import huggingface_hub
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION, files_metadata=True, token=False)
    if info.sha != DATASET_REVISION:
        raise ValueError("Hub returned a different revision")
    files = []
    for sibling in info.siblings:
        if SHARD_RE.fullmatch(sibling.rfilename):
            entry = {"path": sibling.rfilename, "bytes": sibling.size}
            lfs_hash = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
            if lfs_hash:
                entry["expected_sha256"] = lfs_hash
            files.append(entry)
    return validate_inventory(files), huggingface_hub.__version__


def prepare_source(args, raw_root):
    files, hub_version = hub_inventory()
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
                          local_dir=str(raw_root), allow_patterns=[e["path"] for e in files] + ["README.md"],
                          max_workers=args.workers, token=False)
    for entry in files:
        path = raw_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size shard: {path}")
    adi.atomic_write(raw_root / "download_record.json", adi.json_bytes({
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "files": files,
        "huggingface_hub_version": hub_version, "file_sizes_verified": True,
        "test_files_downloaded_by_this_script": False, "checked_at_utc": adi.now()}))
    return files


def make_clip_identity(shard_path, row_index):
    key = f"{DATASET_ID}\n{DATASET_REVISION}\n{shard_path}\n{row_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def channel_stats(payload):
    """Whether a multichannel clip is duplicated mono; the downmix policy depends on it."""
    audio, _ = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    if audio.shape[1] == 1:
        return {"channels_identical": None, "channel_max_abs_difference": None}
    difference = float(np.abs(audio - audio[:, :1]).max(initial=0))
    return {"channels_identical": difference == 0.0, "channel_max_abs_difference": difference}


def write_audio(destination, payload):
    expected = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if destination.stat().st_size != len(payload) or adi.sha256_file(destination) != expected:
            raise OSError(f"Existing extracted audio differs: {destination}; inspect before replacing it")
        return
    adi.atomic_write(destination, payload)
    if adi.sha256_file(destination) != expected:
        raise OSError(f"Written audio failed hash verification: {destination}")


def extract_row(row, config, shard_path, row_index, audio_root):
    """Return (manifest record, exclusion or None, audit-only fields that never enter the manifest)."""
    clip_id = make_clip_identity(shard_path, row_index)
    audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}
    payload = audio.get("bytes")
    original = audio.get("path") if isinstance(audio.get("path"), str) else None
    published = row.get("duration")
    relative = str(PurePosixPath("dev") / config / f"{clip_id}.wav")
    record = {"clip_id": clip_id, "audio_path": relative, "label": CONFIG_TO_LABEL[config], "split": "dev",
              "source_id": None, "source_id_method": "unknown", "speaker_id": None, "domain": DOMAIN,
              "source_dataset": DATASET_ID, "source_revision": DATASET_REVISION, "source_config": config,
              "source_shard": shard_path, "source_row": row_index,
              "seg_id": row.get("seg_id") if isinstance(row.get("seg_id"), str) else None,
              "original_path": original,
              "published_duration_sec": float(published) if isinstance(published, (int, float)) else None}
    audit_only = {"transcription": row.get("transcription"), "gender": row.get("gender")}
    try:
        inspection = adi.inspect_audio_bytes(payload)
        inspection.update(channel_stats(payload))
    except adi.AudioAuditError as error:
        if isinstance(payload, (bytes, bytearray)) and payload:
            write_audio(audio_root / relative, bytes(payload))  # models must fail on it explicitly
        else:
            record["audio_path"] = None
        empty = dict.fromkeys(["duration_sec", "sample_rate_hz", "channel_count", "frame_count", "audio_format",
                               "audio_subtype", "audio_sha256", "pcm_sha256", "peak_abs", "rms",
                               "near_full_scale_fraction", "encoded_bytes", "channels_identical",
                               "channel_max_abs_difference"])
        record.update(empty, preparation_status="audio_error")
        exclusion = {"clip_id": clip_id, "label": record["label"], "source_shard": shard_path,
                     "source_row": row_index, "seg_id": record["seg_id"], "reason": str(error),
                     "status": "audio_error", "kept_in_denominator": True}
        return record, exclusion, audit_only
    write_audio(audio_root / relative, bytes(payload))
    record.update(inspection, preparation_status="ok")
    return record, None, audit_only


def audit_shard(path, entry, audio_root, batch_size):
    match = SHARD_RE.fullmatch(entry["path"])
    config = match["config"]
    adi.progress(f"Hashing {entry['path']}")
    source_hash = adi.sha256_file(path)
    if entry.get("expected_sha256") and source_hash != entry["expected_sha256"]:
        raise ValueError(f"Source content differs from Hub LFS hash: {entry['path']}")
    records, exclusions, audit_rows = [], [], []
    with pq.ParquetFile(path) as parquet:
        missing = set(COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{entry['path']}: missing columns {sorted(missing)}")
        expected_rows = parquet.metadata.num_rows
        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
            for row in batch.to_pylist():
                record, exclusion, audit_only = extract_row(row, config, entry["path"], row_index, audio_root)
                records.append(record)
                audit_rows.append(audit_only)
                if exclusion:
                    exclusions.append(exclusion)
                row_index += 1
                if row_index % adi.PROGRESS_EVERY == 0:
                    adi.progress(f"{entry['path']}: {row_index}/{expected_rows} rows, {len(exclusions)} audio errors")
    if row_index != expected_rows:
        raise ValueError(f"Parquet row count mismatch: {entry['path']}")
    inventory = {**entry, "sha256": source_hash, "rows": row_index,
                 "hub_hash_verified": bool(entry.get("expected_sha256"))}
    return records, exclusions, audit_rows, inventory


def exact_duplicates(records):
    findings = []
    for kind, field in [("encoded_exact", "audio_sha256"), ("decoded_exact", "pcm_sha256")]:
        groups = defaultdict(list)
        for row in records:
            if row[field] is not None:
                groups[row[field]].append(row)
        for value, members in sorted(groups.items()):
            if len(members) > 1:
                prefixes = {seg_prefix(m["seg_id"]) for m in members}
                findings.append({"kind": kind, "value": value,
                                 "members": [{"clip_id": m["clip_id"], "label": m["label"], "seg_id": m["seg_id"],
                                              "duration_sec": m["duration_sec"]} for m in members],
                                 "label_conflict": len({m["label"] for m in members}) > 1,
                                 "same_seg_prefix": len(prefixes) == 1 and None not in prefixes})
    return findings


def adi20_overlap(records, manifest_paths):
    """Exact encoded/PCM matches against ADI-20; resampled or re-encoded copies cannot match."""
    index = {"audio_sha256": defaultdict(list), "pcm_sha256": defaultdict(list)}
    source_ids = defaultdict(list)
    counts = {}
    for path in manifest_paths:
        rows = 0
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                for field in index:
                    if row.get(field):
                        index[field][row[field]].append({"clip_id": row["clip_id"], "split": row["split"],
                                                         "label": row["label"]})
                if row.get("source_id"):
                    source_ids[row["source_id"]].append(row["clip_id"])
        counts[str(Path(path).relative_to(REPO_ROOT)) if Path(path).is_relative_to(REPO_ROOT) else str(path)] = {
            "rows": rows, "sha256": adi.sha256_file(path)}
    findings = []
    for row in records:
        for field, kind in [("audio_sha256", "encoded_exact"), ("pcm_sha256", "decoded_exact")]:
            if row[field] and row[field] in index[field]:
                findings.append({"kind": kind, "clip_id": row["clip_id"], "label": row["label"],
                                 "adi20_matches": index[field][row[field]]})
        inferred = adi.source_metadata(row["original_path"])["source_id"]
        if inferred and inferred in source_ids:
            findings.append({"kind": "filename_youtube_id", "clip_id": row["clip_id"], "label": row["label"],
                             "source_id": inferred, "adi20_clip_count": len(source_ids[inferred])})
    return findings, counts


def seg_prefix(seg_id):
    match = SEG_ID_RE.fullmatch(seg_id or "")
    return int(match["prefix"]) if match else None


def seg_id_analysis(records):
    """Describe seg_id structure. A shared prefix is a hypothesis, not a recovered episode or speaker."""
    per_config, prefix_configs = {}, defaultdict(set)
    seg_counts = Counter((r["source_config"], r["seg_id"]) for r in records)
    for config in CONFIG_TO_LABEL:
        rows = [r for r in records if r["source_config"] == config]
        parsed = [SEG_ID_RE.fullmatch(r["seg_id"] or "") for r in rows]
        prefixes = Counter(int(m["prefix"]) for m in parsed if m)
        for prefix in prefixes:
            prefix_configs[prefix].add(config)
        sizes = Counter(prefixes.values())
        values = sorted(prefixes)
        per_config[config] = {
            "rows": len(rows), "unparsed_seg_ids": sum(m is None for m in parsed),
            "duplicate_seg_ids_within_config": sum(c - 1 for (cfg, _), c in seg_counts.items() if cfg == config and c > 1),
            "distinct_prefixes": len(prefixes), "prefix_min": values[0] if values else None,
            "prefix_max": values[-1] if values else None,
            "clips_per_prefix_histogram": {str(k): sizes[k] for k in sorted(sizes)},
            "max_clips_per_prefix": max(prefixes.values(), default=0),
            "suffix_min": min((int(m["index"]) for m in parsed if m), default=None),
            "suffix_max": max((int(m["index"]) for m in parsed if m), default=None)}
    shared = {p: sorted(c) for p, c in prefix_configs.items() if len(c) > 1}
    return {"pattern": SEG_ID_RE.pattern, "per_config": per_config,
            "prefixes_shared_across_configs": len(shared),
            "examples_shared_across_configs": dict(sorted(shared.items())[:20]),
            "interpretation": "unverified; seg_id prefixes are not treated as episode, scene or speaker IDs"}


def describe(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    return {"min": ordered[0], "p05": pick(0.05), "median": statistics.median(ordered), "p95": pick(0.95),
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def summarize(records, audit_rows):
    ok = [r for r in records if r["preparation_status"] == "ok"]
    durations = [r["duration_sec"] for r in ok]
    stereo = [r for r in ok if r["channel_count"] and r["channel_count"] > 1]
    texts = [a["transcription"] for a in audit_rows]
    text_hashes = Counter(hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
                          for t in texts if isinstance(t, str) and t.strip())
    longest = max(ok, key=lambda r: r["duration_sec"], default=None)
    return {"rows": len(records), "ok": len(ok), "audio_errors": len(records) - len(ok),
            "duration_hours": sum(durations) / 3600, "duration_sec": describe(durations),
            "clips_under_0_5_sec": sum(d < 0.5 for d in durations), "clips_under_1_sec": sum(d < 1 for d in durations),
            "clips_over_10_sec": sum(d > 10 for d in durations), "clips_over_30_sec": sum(d > 30 for d in durations),
            "longest_clip": None if longest is None else {"clip_id": longest["clip_id"],
                                                         "duration_sec": longest["duration_sec"],
                                                         "frame_count": longest["frame_count"],
                                                         "sample_rate_hz": longest["sample_rate_hz"]},
            "sample_rate_counts": dict(Counter(str(r["sample_rate_hz"]) for r in ok)),
            "channel_counts": dict(Counter(str(r["channel_count"]) for r in ok)),
            "format_counts": dict(Counter(f"{r['audio_format']}/{r['audio_subtype']}" for r in ok)),
            "stereo_identical_channels": sum(r["channels_identical"] is True for r in stereo),
            "stereo_distinct_channels": sum(r["channels_identical"] is False for r in stereo),
            "silent_clips": sum(r["peak_abs"] == 0 for r in ok),
            "clips_with_near_full_scale_samples": sum(r["near_full_scale_fraction"] > 0 for r in ok),
            "clips_over_1pct_near_full_scale": sum(r["near_full_scale_fraction"] > 0.01 for r in ok),
            "missing_published_duration": sum(r["published_duration_sec"] is None for r in records),
            "gender_counts_audit_only": dict(Counter(str(a["gender"]) for a in audit_rows)),
            "empty_transcriptions_audit_only": sum(not (isinstance(t, str) and t.strip()) for t in texts),
            "transcription_duplicate_groups_audit_only": sum(c > 1 for c in text_hashes.values()),
            "clips_in_transcription_duplicate_groups_audit_only": sum(c for c in text_hashes.values() if c > 1),
            "encoded_bytes": sum(r["encoded_bytes"] for r in ok)}


def run(args):
    started_at, started = adi.now(), time.monotonic()
    labels = adi.load_labels(args.labels)
    if any(label not in labels for label in CONFIG_TO_LABEL.values()):
        raise ValueError("Casablanca label mapping is not a subset of labels.json")
    root = args.data_root.expanduser().resolve()
    raw_root = (args.raw_dir or root / "raw" / "casablanca" / DATASET_REVISION).expanduser().resolve()
    audio_rel = f"audio/casablanca/{DATASET_REVISION}"
    audio_root = root / audio_rel
    adi.progress("Checking pinned validation inventory")
    files = prepare_source(args, raw_root)
    records, exclusions, audit_rows, inventory = [], [], [], []
    for number, entry in enumerate(files, 1):
        adi.progress(f"Shard {number}/{len(files)}: {entry['path']}")
        shard = audit_shard(raw_root / entry["path"], entry, audio_root, args.batch_size)
        records += shard[0]
        exclusions += shard[1]
        audit_rows += shard[2]
        inventory.append(shard[3])
    counts = Counter(r["source_config"] for r in records)
    if dict(counts) != PUBLISHED_VALIDATION_COUNTS:
        raise ValueError(f"Decoded rows differ from published counts: {dict(counts)}; review before freezing")
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("Duplicate clip IDs")

    code_hash, labels_hash = adi.sha256_file(__file__), adi.sha256_file(args.labels)
    identity = {"dataset_revision": DATASET_REVISION, "pipeline_version": PIPELINE_VERSION,
                "labels_sha256": labels_hash, "inventory": [{"path": e["path"], "sha256": e["sha256"]}
                                                            for e in inventory]}
    split_version = "casablanca-val-v1-" + hashlib.sha256(adi.json_bytes(identity)).hexdigest()[:12]
    support = [label for label in labels if any(r["label"] == label for r in records)]

    manifest_path = args.manifest_dir / "casablanca_validation.jsonl"
    adi.write_jsonl(manifest_path, records)
    manifest_hash = adi.sha256_file(manifest_path)
    extensions = {"source_dataset": "string", "source_revision": "string", "source_config": "string",
                  "source_shard": "string", "source_row": "integer", "seg_id": "string|null",
                  "original_path": "string|null", "published_duration_sec": "number|null",
                  "source_id_method": "string", "preparation_status": "string (ok|audio_error)",
                  "channel_count": "integer|null", "frame_count": "integer|null", "audio_format": "string|null",
                  "audio_subtype": "string|null", "audio_sha256": "string|null", "pcm_sha256": "string|null",
                  "peak_abs": "number|null", "rms": "number|null", "near_full_scale_fraction": "number|null",
                  "encoded_bytes": "integer|null", "channels_identical": "boolean|null",
                  "channel_max_abs_difference": "number|null"}
    metadata = {"schema_version": "1.0", "dataset_version": DATASET_REVISION, "split_version": split_version,
                "split": "dev", "manifest_sha256": manifest_hash, "labels_sha256": labels_hash,
                "audio_root": audio_rel, "audio_root_is_relative_to": "--data-root of prepare_casablanca.py",
                "evaluation_label_sets": {"dev": support}, "selection_split": None,
                "evaluation_role": "frozen_diagnostic_week2_not_for_selection",
                "primary_selection_set": "ADI-20 source development",
                "output_inventory": "all 20 labels; predictions to unsupported labels count as errors",
                "expected_rows": len(records), "ok_rows": len(records) - len(exclusions),
                "audio_error_rows": len(exclusions), "audio_error_rows_in_denominator": True,
                "source_provenance": {"dataset_id": DATASET_ID, "revision": DATASET_REVISION,
                                      "license": "CC BY-NC-ND 4.0", "split_read": "validation only"},
                "label_mapping": {"source": "configuration directory", "configs": CONFIG_TO_LABEL},
                "source_independence_verified": False, "speaker_independence_verified": False,
                "episode_grouping_recovered": False, "checkpoint_pretraining_exposure": "unknown",
                "test_split_used": False, "transcripts_in_manifest": False,
                "pipeline_version": PIPELINE_VERSION, "preparation_code_sha256": code_hash,
                "rules_doc_sha256": adi.sha256_file(RULES_DOC), "extensions": extensions}
    adi.atomic_write(manifest_path.with_suffix(".metadata.json"), adi.json_bytes(metadata))
    freeze = {"split_version": split_version, "dataset_id": DATASET_ID, "dataset_version": DATASET_REVISION,
              "manifest_sha256": manifest_hash, "labels_sha256": labels_hash,
              "label_counts": {label: counts[config] for config, label in CONFIG_TO_LABEL.items()},
              "expected_rows": len(records), "audio_error_clip_ids": [e["clip_id"] for e in exclusions],
              "clip_ids_in_manifest_order": [r["clip_id"] for r in records],
              "clip_ids_sha256": hashlib.sha256("\n".join(r["clip_id"] for r in records).encode()).hexdigest(),
              "inventory": [{k: e[k] for k in ("path", "bytes", "sha256", "rows", "hub_hash_verified")}
                            for e in inventory],
              "pipeline_version": PIPELINE_VERSION, "preparation_code_sha256": code_hash}
    adi.atomic_write(args.manifest_dir / "casablanca_validation_freeze.json", adi.json_bytes(freeze))

    report = args.report_dir
    duplicates = exact_duplicates(records)
    overlap, adi20_inputs = adi20_overlap(records, args.adi20_manifests)
    mismatches = [{"clip_id": r["clip_id"], "label": r["label"], "published_duration_sec": r["published_duration_sec"],
                   "decoded_duration_sec": r["duration_sec"],
                   "difference_sec": r["duration_sec"] - r["published_duration_sec"]}
                  for r in records if r["preparation_status"] == "ok" and r["published_duration_sec"] is not None
                  and abs(r["duration_sec"] - r["published_duration_sec"]) > DURATION_TOLERANCE_SEC]
    seg_analysis = seg_id_analysis(records)
    adi.write_jsonl(report / "exclusions.jsonl", exclusions)
    adi.write_jsonl(report / "duplicates.jsonl", duplicates)
    adi.write_jsonl(report / "adi20_overlap.jsonl", overlap)
    adi.write_jsonl(report / "duration_mismatches.jsonl", mismatches)
    adi.atomic_write(report / "seg_id_analysis.json", adi.json_bytes(seg_analysis))
    per_config = {}
    for config in CONFIG_TO_LABEL:
        selected = [i for i, r in enumerate(records) if r["source_config"] == config]
        per_config[config] = summarize([records[i] for i in selected], [audit_rows[i] for i in selected])
    audit = {"status": "completed_with_audio_errors" if exclusions else "completed",
             "split_version": split_version, "started_at_utc": started_at, "finished_at_utc": adi.now(),
             "runtime_sec": round(time.monotonic() - started, 1),
             "published_validation_counts": PUBLISHED_VALIDATION_COUNTS, "decoded_counts": dict(counts),
             "summary": summarize(records, audit_rows), "per_config": per_config,
             "exact_duplicate_groups": dict(Counter(d["kind"] for d in duplicates)),
             "exact_duplicate_label_conflicts": sum(d["label_conflict"] for d in duplicates),
             "adi20_overlap_findings": dict(Counter(o["kind"] for o in overlap)),
             "adi20_manifests_checked": adi20_inputs,
             "duration_mismatches_over_tolerance": len(mismatches), "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
             "seg_id_prefixes_shared_across_configs": seg_analysis["prefixes_shared_across_configs"],
             "near_duplicate_audit_completed": False, "source_ids_recovered": False,
             "speaker_independence_verified": False, "validation_test_overlap_checked": False,
             "manifest": {"path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_relative_to(REPO_ROOT)
                          else str(manifest_path), "sha256": manifest_hash},
             "processed_shards": inventory,
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__,
                          "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
             "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                          "cpu_count": os.cpu_count()},
             "code_sha256": code_hash, "labels_sha256": labels_hash, "pipeline_version": PIPELINE_VERSION,
             "notes": ["Original WAV bytes preserved; no resampling, trimming, channel mixing or normalization.",
                       "audio_error rows stay in the manifest and the denominator.",
                       "Exact duplicates and ADI-20 exact matches are reported, not removed.",
                       "Transcripts and gender are summarized here only; they are not in the manifest.",
                       "Casablanca test was not downloaded, decoded or hashed."]}
    adi.atomic_write(report / "audit.json", adi.json_bytes(audit))
    print(f"\n{audit['status']}: {len(records)} clips, {len(exclusions)} audio errors", flush=True)
    print(f"SPLIT: {split_version}\nMANIFEST: {manifest_path} ({manifest_hash})\nAUDIT: {report / 'audit.json'}",
          flush=True)
    return audit


def main():
    args = parse_args()
    audit = run(args)
    return 2 if audit["summary"]["audio_errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun to resume (extracted audio is verified, not rewritten).", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
