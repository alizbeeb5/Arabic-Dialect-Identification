#!/usr/bin/env python3
"""Freeze the W1-07 pilot subsets from a prepared ADI-20 Micro manifest release.

Reads the train/dev manifests written by src/data/prepare.py, groups clips that
share an inferred recording ID or identical audio, keeps every group inside one
split, and draws a fixed, class-balanced pilot subset from each split.

Upstream splits are preserved: no clip moves between train and dev. Groups that
cross the upstream boundary are reported and left out of both pilot subsets.

Standard library only. CPU only. Deterministic for a given input and seed.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_VERSION = "w1-07-pilot-v1"
DEFAULT_SEED = 490
DEFAULT_TRAIN_PER_LABEL = 25
DEFAULT_DEV_TOTAL = 200
GROUP_FIELDS = ("source_id", "audio_sha256", "pcm_sha256")
REQUIRED_FIELDS = ("clip_id", "audio_path", "label", "split", "source_id",
                   "speaker_id", "domain", "duration_sec", "sample_rate_hz")


class FreezeError(ValueError):
    """The inputs cannot support a trustworthy split freeze."""


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode("utf-8")


def pretty(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            + "\n").encode("utf-8")


def load_labels(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = config["label_order"]
    if len(labels) != 20 or len(set(labels)) != 20:
        raise FreezeError("labels.json must define 20 unique labels")
    return labels


def read_manifest(manifest_dir, split, labels):
    """Read one prepared split and check it against its sidecar."""
    path = Path(manifest_dir) / f"{split}.jsonl"
    meta_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not meta_path.is_file():
        raise FreezeError(f"Missing {path.name} or {meta_path.name} in {manifest_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    if metadata.get("manifest_sha256") != actual_hash:
        raise FreezeError(f"{path.name} does not match the SHA256 in its sidecar; "
                          "it was edited or truncated after preparation")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise FreezeError(f"{path.name}:{number} lacks {missing}")
        if row["split"] != split:
            raise FreezeError(f"{path.name}:{number} has split={row['split']!r}")
        if row["label"] not in labels:
            raise FreezeError(f"{path.name}:{number} has invalid label {row['label']!r}")
        audio_path = PurePosixPath(row["audio_path"])
        if audio_path.is_absolute() or ".." in audio_path.parts:
            raise FreezeError(f"{path.name}:{number} audio_path escapes audio_root")
        rows.append(row)
    return rows, metadata, actual_hash


class Groups:
    """Union-find over clip IDs; clips sharing any grouping key are joined."""

    def __init__(self, clip_ids):
        self.parent = {clip_id: clip_id for clip_id in clip_ids}

    def find(self, clip_id):
        root = clip_id
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[clip_id] != root:
            self.parent[clip_id], clip_id = root, self.parent[clip_id]
        return root

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            # Smaller ID as root keeps group IDs independent of input order.
            self.parent[max(a, b)] = min(a, b)


def build_groups(rows):
    groups = Groups([row["clip_id"] for row in rows])
    links = Counter()
    for field in GROUP_FIELDS:
        first_seen = {}
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            if value in first_seen:
                groups.union(first_seen[value], row["clip_id"])
                links[field] += 1
            else:
                first_seen[value] = row["clip_id"]
    members = defaultdict(list)
    for row in rows:
        members[groups.find(row["clip_id"])].append(row)
    return {"group:" + sha256_bytes(root.encode())[:16]: rows_in_group
            for root, rows_in_group in members.items()}, links


def draw_key(seed, value):
    """Seeded order that does not depend on Python's random module or row order."""
    return sha256_bytes(f"{FREEZE_VERSION}:{seed}:{value}".encode("utf-8"))


def sample_balanced(rows, group_of, labels, quotas, seed, max_per_group=None):
    """Pick up to quotas[label] clips per label, spreading picks across groups.

    Round-robin: one clip from every group (in seeded group order), then a second
    from every group, and so on. A label drawn from few recordings still fills its
    quota, but no recording supplies a second clip while another has supplied none.
    max_per_group, if set, is a hard cap per group.
    """
    by_label = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_label[row["label"]][group_of[row["clip_id"]]].append(row)
    chosen, shortfall, spread = [], {}, {}
    for label in labels:
        queues = [sorted(members, key=lambda r: draw_key(seed, r["clip_id"]))
                  for _, members in sorted(by_label.get(label, {}).items(),
                                           key=lambda item: draw_key(seed, item[0]))]
        if max_per_group is not None:
            queues = [queue[:max_per_group] for queue in queues]
        picked, depth = [], 0
        while len(picked) < quotas[label] and any(depth < len(q) for q in queues):
            for queue in queues:
                if len(picked) == quotas[label]:
                    break
                if depth < len(queue):
                    picked.append(queue[depth])
            depth += 1
        eligible = sum(len(members) for members in by_label.get(label, {}).values())
        if len(picked) < quotas[label]:
            shortfall[label] = {"wanted": quotas[label], "selected": len(picked),
                                "eligible_clips": eligible}
        per_group = Counter(group_of[r["clip_id"]] for r in picked)
        spread[label] = {"groups_available": len(queues), "groups_used": len(per_group),
                         "max_clips_from_one_group": max(per_group.values(), default=0)}
        chosen.extend(picked)
    chosen.sort(key=lambda r: r["clip_id"])
    return chosen, shortfall, spread


def even_quotas(labels, total):
    base, extra = divmod(total, len(labels))
    return {label: base + (index < extra) for index, label in enumerate(labels)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, required=True,
                        help="Release folder from prepare.py: DATA_ROOT/manifests/adi20_micro/<rev>/<release_id>")
    parser.add_argument("--dev-split", default="dev",
                        help="Prepared split that serves as the in-source development pool (default: dev)")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-label", type=int, default=DEFAULT_TRAIN_PER_LABEL)
    parser.add_argument("--dev-total", type=int, default=DEFAULT_DEV_TOTAL)
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="Optional hard cap on clips from one recording/duplicate group per subset "
                             "(default: no cap; picks are still spread round-robin across groups)")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT,
                        help="Writes data/manifests/ and reports/ under this folder (default: repository root)")
    parser.add_argument("--allow-pilot-source", action="store_true",
                        help="Accept a bounded prepare.py pilot release (outputs marked development-only)")
    args = parser.parse_args(argv)
    for name in ("train_per_label", "dev_total", "max_per_group"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dev_split in ("train", "test"):
        parser.error("--dev-split cannot be train or test; use development data")
    return args


def run(args):
    # Direct Python callers must respect the same boundary as the CLI.
    if args.dev_split in ("train", "test"):
        raise FreezeError("--dev-split cannot be train or test; use development data")
    labels = load_labels(args.labels)
    labels_hash = sha256_file(args.labels)
    train_rows, train_meta, train_hash = read_manifest(args.manifest_dir, "train", labels)
    dev_rows, dev_meta, dev_hash = read_manifest(args.manifest_dir, args.dev_split, labels)
    source_is_pilot = bool(train_meta.get("pilot") or dev_meta.get("pilot"))
    if source_is_pilot and not args.allow_pilot_source:
        raise FreezeError("Source manifests come from a bounded prepare.py pilot. Rerun on the full "
                          "release, or pass --allow-pilot-source for a development-only dry run.")
    for name, metadata in (("train", train_meta), (args.dev_split, dev_meta)):
        if metadata.get("labels_sha256") not in (None, labels_hash):
            raise FreezeError(f"{name} manifest was prepared with a different labels.json")
    if train_meta.get("dataset_version") != dev_meta.get("dataset_version"):
        raise FreezeError("train and dev manifests come from different dataset revisions")

    rows = train_rows + dev_rows
    ids = [row["clip_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise FreezeError("A clip_id appears more than once across train and dev")

    groups, links = build_groups(rows)
    group_of = {row["clip_id"]: gid for gid, members in groups.items() for row in members}
    crossing = {gid: members for gid, members in groups.items()
                if len({row["split"] for row in members}) > 1}
    # Identical audio carrying two different labels is a label error, not just leakage.
    conflicting = set()
    for field in ("audio_sha256", "pcm_sha256"):
        labels_by_hash = defaultdict(set)
        for row in rows:
            if row.get(field) is not None:
                labels_by_hash[row[field]].add(row["label"])
        for row in rows:
            if row.get(field) is not None and len(labels_by_hash[row[field]]) > 1:
                conflicting.add(group_of[row["clip_id"]])

    exclusions = []

    def eligible(split_rows):
        kept = []
        for row in split_rows:
            gid = group_of[row["clip_id"]]
            reason = None
            if gid in crossing:
                reason = "group_crosses_train_dev_boundary"
            elif gid in conflicting:
                reason = "identical_audio_with_conflicting_labels"
            elif row.get("peak_abs") == 0:
                reason = "silent_audio"
            if reason:
                exclusions.append({"clip_id": row["clip_id"], "split": row["split"],
                                   "label": row["label"], "group_id": gid, "reason": reason})
            else:
                kept.append(row)
        return kept

    train_pool, dev_pool = eligible(train_rows), eligible(dev_rows)
    train_pilot, train_short, train_spread = sample_balanced(
        train_pool, group_of, labels, {label: args.train_per_label for label in labels},
        args.seed, args.max_per_group)
    dev_pilot, dev_short, dev_spread = sample_balanced(
        dev_pool, group_of, labels, even_quotas(labels, args.dev_total),
        args.seed + 1, args.max_per_group)

    # The pilots must be subsets of their own upstream split, row for row.
    source_rows = {row["clip_id"]: row for row in rows}
    for name, chosen, split in (("train_pilot", train_pilot, "train"), ("dev_pilot", dev_pilot, args.dev_split)):
        for row in chosen:
            if source_rows.get(row["clip_id"]) != row or row["split"] != split:
                raise FreezeError(f"Internal check failed: {name} row {row['clip_id']} is not an "
                                  f"unchanged row of upstream {split}")
    train_groups = {group_of[r["clip_id"]] for r in train_pilot}
    dev_groups = {group_of[r["clip_id"]] for r in dev_pilot}
    if train_groups & dev_groups or {r["clip_id"] for r in train_pilot} & {r["clip_id"] for r in dev_pilot}:
        raise FreezeError("Internal check failed: pilot subsets share a group")

    train_bytes, dev_bytes = jsonl_bytes(train_pilot), jsonl_bytes(dev_pilot)
    identity = {"freeze_version": FREEZE_VERSION, "dataset_version": train_meta.get("dataset_version"),
                "source_release": train_meta.get("split_version"),
                "source_manifest_sha256": {"train": train_hash, args.dev_split: dev_hash},
                "labels_sha256": labels_hash, "seed": args.seed,
                "train_per_label": args.train_per_label, "dev_total": args.dev_total,
                "max_per_group": args.max_per_group, "dev_split": args.dev_split,
                "train_pilot_sha256": sha256_bytes(train_bytes),
                "dev_pilot_sha256": sha256_bytes(dev_bytes)}
    split_hash = sha256_bytes(canonical(identity))
    split_version = f"{FREEZE_VERSION}-{split_hash[:12]}"

    manifest_out = args.output_root / "data" / "manifests"
    report_out = args.output_root / "reports"
    manifest_out.mkdir(parents=True, exist_ok=True)
    report_out.mkdir(parents=True, exist_ok=True)

    def sidecar(name, payload, chosen, role):
        return {"schema_version": "1.0", "dataset_version": train_meta.get("dataset_version"),
                "split_version": split_version, "split_sha256": split_hash,
                "manifest_sha256": sha256_bytes(payload), "labels_sha256": labels_hash,
                "audio_root": train_meta.get("audio_root"),
                "audio_root_relative_to_data_root": train_meta.get("audio_root_relative_to_data_root"),
                # Keyed by the rows' split value ("train"/"dev"), which is what the evaluator looks up.
                "evaluation_label_sets": {("train" if role == "train" else args.dev_split):
                                          sorted({r["label"] for r in chosen}, key=labels.index)},
                "selection_split": "dev_pilot" if role == "dev" else None,
                "role": role, "clips": len(chosen),
                "label_counts": {label: sum(r["label"] == label for r in chosen) for label in labels},
                "derived_from": {"manifest": f"{'train' if role == 'train' else args.dev_split}.jsonl",
                                 "sha256": train_hash if role == "train" else dev_hash,
                                 "release": train_meta.get("split_version")},
                "source_provenance": train_meta.get("source_provenance"),
                "domain_relation": "in_source (ADI-20 Micro); not out-of-domain",
                "source_independence_verified": False, "speaker_independence_verified": False,
                "development_only": source_is_pilot}

    outputs = {}
    for name, payload, chosen, role in (("train_pilot", train_bytes, train_pilot, "train"),
                                        ("dev_pilot", dev_bytes, dev_pilot, "dev")):
        path = manifest_out / f"{name}.jsonl"
        path.write_bytes(payload)
        path.with_suffix(".metadata.json").write_bytes(pretty(sidecar(name, payload, chosen, role)))
        outputs[name] = str(path)

    freeze = {**identity, "split_version": split_version, "split_sha256": split_hash,
              "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "code_sha256": sha256_file(__file__), "development_only": source_is_pilot,
              "train_pilot_ids": [r["clip_id"] for r in train_pilot],
              "dev_pilot_ids": [r["clip_id"] for r in dev_pilot]}
    (manifest_out / "split_freeze.json").write_bytes(pretty(freeze))

    def shared_keys(members):
        counts = Counter((f, r[f]) for r in members for f in GROUP_FIELDS if r.get(f) is not None)
        return sorted(f"{field}={value}" for (field, value), n in counts.items() if n > 1)

    crossing_rows = [{"group_id": gid, "keys": shared_keys(members),
                      "members": [{"clip_id": r["clip_id"], "split": r["split"], "label": r["label"]}
                                  for r in sorted(members, key=lambda r: r["clip_id"])]}
                     for gid, members in sorted(crossing.items())]
    (report_out / "split_cross_groups.jsonl").write_bytes(jsonl_bytes(crossing_rows))
    (report_out / "split_exclusions.jsonl").write_bytes(
        jsonl_bytes(sorted(exclusions, key=lambda e: e["clip_id"])))

    stats = {"labels": labels, "source_is_pilot": source_is_pilot, "split_version": split_version,
             "split_hash": split_hash, "args": args, "train_meta": train_meta,
             "train_rows": train_rows, "dev_rows": dev_rows, "groups": groups, "links": links,
             "crossing": crossing, "conflicting": conflicting, "exclusions": exclusions,
             "train_pool": train_pool, "dev_pool": dev_pool, "train_pilot": train_pilot,
             "dev_pilot": dev_pilot, "train_short": train_short, "dev_short": dev_short,
             "train_spread": train_spread, "dev_spread": dev_spread,
             "train_hash": train_hash, "dev_hash": dev_hash, "labels_hash": labels_hash}
    (report_out / "split_audit.md").write_text(render_report(stats), encoding="utf-8")
    print(f"Frozen {split_version}: {len(train_pilot)} train-pilot, {len(dev_pilot)} dev-pilot clips")
    print(f"Wrote {manifest_out} and {report_out / 'split_audit.md'}")
    return freeze


def table(labels, columns):
    head = "| Label | " + " | ".join(name for name, _ in columns) + " |"
    rule = "|---|" + "---:|" * len(columns)
    body = [f"| {label} | " + " | ".join(str(counts.get(label, 0)) for _, counts in columns) + " |"
            for label in labels]
    total = "| **Total** | " + " | ".join(str(sum(counts.values())) for _, counts in columns) + " |"
    return "\n".join([head, rule, *body, total])


def render_report(s):
    a, labels = s["args"], s["labels"]
    count = lambda rows: Counter(r["label"] for r in rows)
    all_rows = s["train_rows"] + s["dev_rows"]
    unknown_source = sum(r["source_id"] is None for r in all_rows)
    multi = sum(len(m) > 1 for m in s["groups"].values())
    reasons = Counter(e["reason"] for e in s["exclusions"])
    exclusion_rows = ("\n".join(f"| {reason} | {n} |" for reason, n in sorted(reasons.items()))
                      or "| (none) | 0 |")
    cap_note = (f" A hard cap of {a.max_per_group} clip(s) per group also applies."
                if a.max_per_group else "")
    known = lambda rows, label: sum(r["label"] == label and r["source_id"] is not None for r in rows)
    coverage = "\n".join(
        f"| {label} | {known(s['train_rows'], label)} / {sum(r['label'] == label for r in s['train_rows'])} "
        f"| {known(s['dev_rows'], label)} / {sum(r['label'] == label for r in s['dev_rows'])} |"
        for label in labels)
    no_source = [label for label in labels
                 if not known(s["train_rows"], label) and not known(s["dev_rows"], label)]
    testable = [label for label in labels
                if known(s["train_rows"], label) and known(s["dev_rows"], label)]
    one_sided = [label for label in labels if label not in testable and label not in no_source]
    spread = "\n".join(
        f"| {label} | {s['train_spread'][label]['groups_used']} / {s['train_spread'][label]['groups_available']} "
        f"| {s['train_spread'][label]['max_clips_from_one_group']} "
        f"| {s['dev_spread'][label]['groups_used']} / {s['dev_spread'][label]['groups_available']} "
        f"| {s['dev_spread'][label]['max_clips_from_one_group']} |"
        for label in labels)
    verification = s["train_meta"].get("stored_audio_verification")
    shortfalls = [f"- {split} {label}: wanted {v['wanted']}, selected {v['selected']} "
                  f"({v['eligible_clips']} eligible clips)"
                  for split, short in (("train", s["train_short"]), ("dev", s["dev_short"]))
                  for label, v in short.items()]
    banner = ("> **DEVELOPMENT-ONLY DRY RUN.** The source manifests are a bounded prepare.py pilot. "
              "Do not use these files for experiments; rerun on the full release.\n\n"
              if s["source_is_pilot"] else "")
    command = (f"python src/data/freeze_splits.py --manifest-dir {a.manifest_dir} --dev-split {a.dev_split} "
               f"--seed {a.seed} --train-per-label {a.train_per_label} --dev-total {a.dev_total}"
               + (f" --max-per-group {a.max_per_group}" if a.max_per_group else "")
               + (" --allow-pilot-source" if a.allow_pilot_source else ""))
    return f"""# W1-07 split audit

{banner}**Split version:** `{s['split_version']}`
**Split SHA256:** `{s['split_hash']}`
**Source release:** `{s['train_meta'].get('split_version')}` (dataset revision `{s['train_meta'].get('dataset_version')}`)
**Labels SHA256:** `{s['labels_hash']}`

Exact command:

```bash
{command}
```

## 1. Split policy

- The upstream ADI-20 Micro split is preserved. `train` comes from upstream train; `{a.dev_split}` comes from the upstream validation split as mapped by `prepare.py`. No clip was moved between them.
- The development pool is **in-source**: same collection as training. It is not out-of-domain and must not be reported as such.
- No final-test data was read. Neither NADI 2026 test nor Casablanca test is an input to this script, so neither can influence tuning, pilot selection or model-generated labels.
- Grouping metadata (source IDs, hashes) was used only to build and audit splits. It is not a model input.

## 2. Grouping and separation

Clips were joined into one group whenever they shared any of: the filename-inferred `source_id`, the encoded-audio SHA256, or the decoded-PCM SHA256. Groups were formed transitively.

| Check | Result |
|---|---:|
| Clips read (train / {a.dev_split}) | {len(s['train_rows'])} / {len(s['dev_rows'])} |
| Clips with unknown `source_id` | {unknown_source} |
| Clips with known `speaker_id` | {sum(r['speaker_id'] is not None for r in all_rows)} |
| Groups | {len(s['groups'])} |
| Groups with more than one clip | {multi} |
| Links via source_id / audio_sha256 / pcm_sha256 | {s['links'].get('source_id', 0)} / {s['links'].get('audio_sha256', 0)} / {s['links'].get('pcm_sha256', 0)} |
| Groups crossing the train/{a.dev_split} boundary | {len(s['crossing'])} ({sum(len(m) for m in s['crossing'].values())} clips) |
| Identical-audio groups with conflicting labels | {len(s['conflicting'])} |

Groups crossing the boundary are listed in `reports/split_cross_groups.jsonl`. All of their clips were excluded from **both** pilot subsets. The upstream split itself still contains them; full-split training must decide separately whether to drop them.

**Limitations.** `source_id` is inferred from filenames and is unverified. No speaker IDs are available, so **speaker independence is not claimed**. Clips with unknown `source_id` are singleton groups unless an exact duplicate links them; recording overlap among those clips cannot be ruled out. Near-duplicate audio (re-encoded or trimmed copies) was not checked; only exact encoded and decoded matches were.

### Recording-ID coverage by dialect

A recording ID can only be inferred when the upstream filename has a YouTube ID. Clips with a known ID / all clips:

| Label | Train | {a.dev_split.capitalize()} |
|---|---:|---:|
{coverage}

A shared-recording check between train and {a.dev_split} is only possible where **both** splits have IDs: {", ".join(testable) if testable else "no dialect"}.
- IDs on one side only: {", ".join(one_sided) if one_sided else "none"}. Nothing on the other side to compare against.
- No IDs in either split: {", ".join(no_source) if no_source else "none"}.

For every dialect outside the testable list, recording separation rests only on the upstream split and the exact-duplicate checks; it is not verified.

## 3. Pilot exclusions

| Reason | Clips |
|---|---:|
{exclusion_rows}

Per-clip records: `reports/split_exclusions.jsonl`.

## 4. Sampling procedure

1. Start from the eligible pools above.
2. For each label, sort its clips by `SHA256("{FREEZE_VERSION}:<seed>:<clip_id>")`. This order depends only on the seed and clip IDs, not on Python's random module or the manifest row order.
3. Order that label's groups by the same seeded hash. Take clips round-robin: the first clip of every group, then the second clip of every group, and so on, until the quota is met. No group supplies a second clip while another eligible group has supplied none.{cap_note}
4. Train pilot: seed {a.seed}, {a.train_per_label} clips per label. Dev pilot: seed {a.seed + 1}, {a.dev_total} clips spread evenly over the 20 labels.

The exact IDs, seeds and hashes are recorded in `data/manifests/split_freeze.json`.

## 5. Class counts

{table(labels, [('Train pool', count(s['train_pool'])), ('Train pilot', count(s['train_pilot'])), (a.dev_split.capitalize() + ' pool', count(s['dev_pool'])), ('Dev pilot', count(s['dev_pilot']))])}

### Shortfalls

""" + ("\n".join(shortfalls) if shortfalls else "None: every label met its quota.") + f"""

### Spread across recording groups

Groups used / groups available, and the most clips taken from any one group. Unknown-source clips are singleton groups, so for those dialects every clip counts as its own group.

| Label | Train groups | Train max/group | Dev groups | Dev max/group |
|---|---:|---:|---:|---:|
{spread}

## 6. Out-of-domain source

No separate-source development set is part of this freeze. Casablanca validation is the planned target-domain development set (`docs/evaluation_access.md`), but it has not been ingested or audited yet. **Week 2 task:** ingest Casablanca validation, recover episode grouping where possible, check recording and exact-duplicate overlap against ADI-20 training, and freeze it as the out-of-domain development set. Until then, every result on `dev_pilot` is in-source.

## 7. Acceptance checks

- [x] `train_pilot` is a subset of upstream train and `dev_pilot` a subset of upstream {a.dev_split}: every row was checked to be an unchanged row of its own upstream manifest.
- [x] Pilot subsets are disjoint by clip ID and by group (recording ID or exact-duplicate audio). Checked by the script before writing.
- [x] Class counts, exclusions, grouping limitations and the exact sampling procedure are recorded above.
- [x] The final test is excluded from tuning, pilot selection and model-generated training labels.
- [ ] A different teammate reviews this report and the generated manifests.

## 8. Input integrity

| File | SHA256 (verified against its sidecar) |
|---|---|
| train.jsonl | `{s['train_hash']}` |
| {a.dev_split}.jsonl | `{s['dev_hash']}` |

Stored-audio verification used by the preparation run: `{verification}`.""" + (
        " Size checks cannot detect same-size corruption; rerun preparation with `--verification sha256` before final evaluation." if verification == "size" else "") + """
"""


def main(argv=None):
    try:
        run(parse_args(argv))
    except FreezeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
