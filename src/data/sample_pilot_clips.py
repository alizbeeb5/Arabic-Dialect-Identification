#!/usr/bin/env python3
"""Stream a handful of pinned ADI20 training rows for encoder smoke runs.

prepare.py downloads all 44 shards (~21.8 GB) even with --max-clips. This reads
only the first Parquet row group of selected train shards over HTTP and passes
rows through prepare.extract_clip, so clip IDs, WAV paths and manifest rows are
identical to the official release. Shards are sorted by dialect, so spacing the
selected shards spreads the sample across dialects.

The output is a pilot sample, not an audited release: it has no exclusion,
overlap or full-shard hash audit. Use the prepared release when it is available.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402

TRAIN_SHARDS = 38


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=prepare.REPO_ROOT / "configs/labels.json")
    parser.add_argument("--shard-step", type=int, default=4,
                        help="Use train shards 0, step, 2*step, ...")
    parser.add_argument("--clips-per-shard", type=int, default=2)
    args = parser.parse_args(argv)
    if args.shard_step <= 0 or not 0 < args.clips_per_shard <= 100:
        parser.error("--shard-step must be positive and --clips-per-shard in 1..100")
    return args


def main(argv=None):
    args = parse_args(argv)
    from huggingface_hub import HfFileSystem

    labels = set(json.loads(args.labels.read_text(encoding="utf-8"))["label_order"])
    rev = prepare.DATASET_REVISION
    audio_root = args.data_root / "audio/adi20_micro" / rev
    out_root = args.data_root / "manifests/adi20_micro" / rev / "pilot-sample"
    fs = HfFileSystem()
    records = []
    for index in range(0, TRAIN_SHARDS, args.shard_step):
        shard = f"train-{index:05d}-of-{TRAIN_SHARDS:05d}.parquet"
        prepare.progress(f"Reading first row group of {shard}")
        with fs.open(f"datasets/{prepare.DATASET_ID}@{rev}/data/{shard}", "rb") as stream:
            rows = pq.ParquetFile(stream).read_row_group(0, columns=["audio", "dialect"]).to_pylist()
        # Row group 0 starts at row 0, so the position is the source row index.
        for row_index, row in enumerate(rows[:args.clips_per_shard]):
            records.append(prepare.extract_clip(row, "train", shard, row_index, audio_root, labels))

    manifest = out_root / "train.jsonl"
    prepare.write_jsonl(manifest, records)
    metadata = {
        "kind": "pilot_sample_not_audited", "dataset_id": prepare.DATASET_ID,
        "dataset_revision": rev, "audio_root": str(audio_root), "split": "train",
        "clip_count": len(records), "shard_step": args.shard_step,
        "clips_per_shard": args.clips_per_shard,
        "labels": sorted({row["label"] for row in records}),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    prepare.atomic_write(out_root / "train.metadata.json", prepare.json_bytes(metadata))
    prepare.progress(f"Wrote {len(records)} clips to {manifest}")


if __name__ == "__main__":
    main()
