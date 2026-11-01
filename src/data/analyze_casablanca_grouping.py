#!/usr/bin/env python3
"""Infer episode and parent-segment groups for the frozen Casablanca validation manifest.

Reads only the committed manifest (no audio). Casablanca original_path names look like
"<episode file title>_<start>_<end>_<seg_id>.wav". The groups are filename inferences for
auditing; they are not verified speakers, and they never enter model inputs.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare as adi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_RE = re.compile(r"(?P<episode>.+)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)_(?P<seg_id>\d+_\d+)\.wav")
METHOD = "inferred_from_original_path_not_verified"


def parse_original_path(original_path):
    match = PATH_RE.fullmatch(original_path or "")
    if not match:
        return None
    return {"episode_file": match["episode"], "window_start_sec": float(match["start"]),
            "window_end_sec": float(match["end"]), "seg_id": match["seg_id"]}


def group_id(kind, *parts):
    return f"casablanca-{kind}:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def analyze(rows):
    assignments, unparsed = [], []
    for row in rows:
        parsed = parse_original_path(row["original_path"])
        if parsed is None or parsed["seg_id"] != row["seg_id"]:
            unparsed.append(row["clip_id"])
            continue
        prefix = row["seg_id"].split("_")[0]
        episode = group_id("episode", row["source_config"], parsed["episode_file"])
        assignments.append({"clip_id": row["clip_id"], "label": row["label"], "episode_group": episode,
                            "parent_segment_group": group_id("segment", episode, prefix),
                            "episode_file": parsed["episode_file"], "window_start_sec": parsed["window_start_sec"],
                            "window_end_sec": parsed["window_end_sec"], "method": METHOD})
    episodes = defaultdict(list)
    segments = defaultdict(list)
    for item in assignments:
        episodes[item["episode_group"]].append(item)
        segments[item["parent_segment_group"]].append(item)
    windows = defaultdict(set)
    for item in assignments:
        windows[item["parent_segment_group"]].add((item["window_start_sec"], item["window_end_sec"]))
    per_label = {}
    for label in sorted({a["label"] for a in assignments}):
        sizes = [len(v) for v in episodes.values() if v[0]["label"] == label]
        per_label[label] = {"clips": sum(sizes), "episode_groups": len(sizes), "max_clips_per_episode": max(sizes),
                            "median_clips_per_episode": sorted(sizes)[len(sizes) // 2],
                            "largest_episode_share": round(max(sizes) / sum(sizes), 4)}
    summary = {"method": METHOD, "rows": len(rows), "parsed": len(assignments), "unparsed_clip_ids": unparsed,
               "episode_groups": len(episodes), "parent_segment_groups": len(segments),
               "episode_groups_spanning_labels": sum(len({a["label"] for a in v}) > 1 for v in episodes.values()),
               "parent_segments_with_one_window": sum(len(w) == 1 for w in windows.values()),
               "clips_sharing_a_parent_segment": sum(len(v) for v in segments.values() if len(v) > 1),
               "clips_per_episode_histogram": dict(sorted(Counter(len(v) for v in episodes.values()).items())),
               "per_label": per_label,
               "notes": ["The filename window is the parent segment span; sub-segment offsets are not given.",
                         "Episode files of one TV series are separate groups; series-level grouping is not derived.",
                         "Speakers are unknown; episodes of one series likely share actors.",
                         "Groups are for auditing only and must not enter model features or prompts."]}
    return summary, assignments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/casablanca_validation.jsonl")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports/casablanca_validation_audit")
    args = parser.parse_args(argv)
    metadata = json.loads(args.manifest.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != adi.sha256_file(args.manifest):
        raise ValueError("Manifest does not match its metadata sidecar")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary, assignments = analyze(rows)
    summary.update(manifest_sha256=metadata["manifest_sha256"], split_version=metadata["split_version"],
                   code_sha256=adi.sha256_file(__file__))
    adi.atomic_write(args.report_dir / "episode_groups.json", adi.json_bytes(summary))
    adi.write_jsonl(args.report_dir / "episode_groups.jsonl", assignments)
    print(f"{summary['parsed']}/{summary['rows']} clips grouped into {summary['episode_groups']} episode files "
          f"and {summary['parent_segment_groups']} parent segments")
    return 0 if not summary["unparsed_clip_ids"] else 1


if __name__ == "__main__":
    sys.exit(main())
