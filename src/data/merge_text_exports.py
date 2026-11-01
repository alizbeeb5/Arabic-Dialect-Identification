#!/usr/bin/env python3
"""W2-02 (EEC-24): combine the W1-13 500-clip and W2-02 1,500-clip transcript exports.

Both inputs are validated with the same loader the text model uses
(src/models/transcript_tfidf.load_export), against their own manifests. The
output is one export for train_scaling_2000 in manifest order, with records
copied unchanged. No transcription, filtering or relabelling happens here.

Usage:
  python src/data/merge_text_exports.py \
    --original-export TRAIN_EXPORT_500 --new-export NEW_EXPORT_1500 --output-dir OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.transcript_tfidf import load_export  # noqa: E402
from src.experiments.jais_pilot import write_jsonl  # noqa: E402
from src.models.jais_adapter import atomic_json, read_json, sha256_file  # noqa: E402

M = ROOT / "data/manifests"
LABELS = ROOT / "configs/labels.json"
PARTS = {
    "original_500": (M / "train_pilot.jsonl", M / "train_pilot.metadata.json"),
    "new_1500": (M / "train_scaling_new_1500.jsonl", M / "train_scaling_new_1500.metadata.json"),
}
TARGET = (M / "train_scaling_2000.jsonl", M / "train_scaling_2000.metadata.json")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-export", type=Path, required=True)
    parser.add_argument("--new-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new output directory")

    records, targets, sources = {}, {}, {}
    for part, export in (("original_500", args.original_export), ("new_1500", args.new_export)):
        manifest, metadata = PARTS[part]
        part_records, part_targets, _, identity = load_export(export, manifest, metadata, LABELS, "train")
        overlap = set(records) & {r["clip_id"] for r in part_records}
        if overlap:
            raise SystemExit(f"{len(overlap)} clip IDs appear in both exports")
        for record in part_records:
            records[record["clip_id"]] = record
        targets.update(part_targets)
        summary = read_json(Path(export) / "summary.json")
        sources[part] = {
            "export_dir": str(export), "identity": identity, "expected": summary["expected"],
            "transcript_counts": summary["transcript_counts"], "refused": summary["refused"],
            "truncated": summary["truncated"], "truncation_unknown": summary["truncation_unknown"],
            "resource_record": summary.get("resource_record"),
        }

    manifest, metadata = TARGET
    rows = read_rows(manifest)
    if {r["clip_id"] for r in rows} != set(records):
        raise SystemExit("Merged IDs do not equal the train_scaling_2000 manifest")
    meta = read_json(metadata)
    ordered = [records[r["clip_id"]] for r in rows]
    for r in rows:
        if targets[r["clip_id"]] != r["label"]:
            raise SystemExit("Target disagrees with the 2,000 manifest")

    out = args.output_dir
    out.mkdir(parents=True)
    write_jsonl(out / "transcripts.jsonl", ordered)
    write_jsonl(out / "targets.jsonl", [{"clip_id": r["clip_id"], "label": r["label"]} for r in rows])
    summary = {
        "schema_version": "1.0", "split": "train", "expected": len(rows),
        "manifest_sha256": sha256_file(manifest), "metadata_sha256": sha256_file(metadata),
        "labels_sha256": sha256_file(LABELS), "split_version": meta["split_version"],
        "dataset_version": meta["dataset_version"],
        "transcript_counts": dict(Counter(r["transcript_status"] for r in ordered)),
        "refused": sum(r["refused"] is True for r in ordered),
        "truncated": sum(r["truncated"] is True for r in ordered),
        "truncation_unknown": sum(r["truncated"] is None for r in ordered),
        "kind": "merged_export_no_new_inference",
        "merge_code_sha256": sha256_file(__file__),
        "sources": sources,
    }
    for name in ("transcripts.jsonl", "targets.jsonl"):
        summary[name + "_sha256"] = sha256_file(out / name)
    atomic_json(out / "summary.json", summary)
    print(json.dumps({"output_dir": str(out), "expected": len(rows),
                      "transcript_counts": summary["transcript_counts"],
                      "by_source": {k: v["transcript_counts"] for k, v in sources.items()}}, indent=2))


if __name__ == "__main__":
    main()
