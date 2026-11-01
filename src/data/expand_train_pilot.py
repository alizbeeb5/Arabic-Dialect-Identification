#!/usr/bin/env python3
"""W2-02 (EEC-24): build the nested 2,000-clip training manifest for text scaling.

Reruns the frozen W1-07 selection (src/data/freeze_splits.py, same seed, same
eligibility rules, same audited full-release manifests) with 100 clips per label
instead of 25. The round-robin draw order does not depend on the quota, so the
first 25 picks per label are exactly the original pilot: the 2,000 IDs strictly
contain the original 500 and the 200 dev IDs are unchanged. Both properties are
checked here and the script fails if either does not hold. No dev performance is
read at any point.

Writes (under --output-root, default repository root):
  data/manifests/train_scaling_2000.jsonl (+ .metadata.json)  all 2,000 rows
  data/manifests/train_scaling_new_1500.jsonl                 only the new rows to transcribe
The frozen W1-07 files (train_pilot.jsonl, dev_pilot.jsonl, split_freeze.json) are not touched.

Reproduce:
  python src/data/expand_train_pilot.py --manifest-dir src/data/manifests
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_splits as fs  # noqa: E402

TARGET_PER_LABEL = 100
EXPANSION_VERSION = "w2-02-text-scaling-v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", type=Path, default=fs.REPO_ROOT / "src/data/manifests",
                        help="Audited full-release train/dev manifests used by W1-07")
    parser.add_argument("--frozen", type=Path,
                        default=fs.REPO_ROOT / "data/manifests/split_freeze.json",
                        help="The W1-07 split_freeze.json to extend")
    parser.add_argument("--labels", type=Path, default=fs.REPO_ROOT / "configs/labels.json")
    parser.add_argument("--output-root", type=Path, default=fs.REPO_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    labels = fs.load_labels(args.labels)

    with tempfile.TemporaryDirectory() as tmp:
        fs.run(fs.parse_args([
            "--manifest-dir", str(args.manifest_dir), "--labels", str(args.labels),
            "--seed", str(frozen["seed"]), "--dev-total", str(frozen["dev_total"]),
            "--train-per-label", str(TARGET_PER_LABEL), "--output-root", tmp]))
        out = Path(tmp) / "data/manifests"
        expanded = json.loads((out / "split_freeze.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (out / "train_pilot.jsonl").read_text(encoding="utf-8").splitlines() if line]

    # Guards: same source, nested training IDs, identical dev IDs, balanced, train-only.
    if expanded["source_manifest_sha256"] != frozen["source_manifest_sha256"]:
        raise SystemExit("Source manifests differ from the ones W1-07 was frozen from")
    if expanded["labels_sha256"] != frozen["labels_sha256"]:
        raise SystemExit("labels.json differs from the W1-07 freeze")
    original = set(frozen["train_pilot_ids"])
    ids = [r["clip_id"] for r in rows]
    if len(ids) != len(set(ids)) or not original <= set(ids):
        raise SystemExit("Expanded set does not strictly contain the original 500 training IDs")
    if expanded["dev_pilot_ids"] != frozen["dev_pilot_ids"]:
        raise SystemExit("Dev pilot IDs changed")
    if set(ids) & set(frozen["dev_pilot_ids"]) or any(r["split"] != "train" for r in rows):
        raise SystemExit("Expanded set contains non-training clips")
    counts = Counter(r["label"] for r in rows)
    if any(counts[label] != TARGET_PER_LABEL for label in labels):
        raise SystemExit(f"Unbalanced expansion: {dict(counts)}")

    new_rows = [r for r in rows if r["clip_id"] not in original]
    full_bytes, new_bytes = fs.jsonl_bytes(rows), fs.jsonl_bytes(new_rows)
    manifest_out = args.output_root / "data/manifests"
    manifest_out.mkdir(parents=True, exist_ok=True)
    (manifest_out / "train_scaling_2000.jsonl").write_bytes(full_bytes)
    (manifest_out / "train_scaling_new_1500.jsonl").write_bytes(new_bytes)

    # Sidecars follow the train_pilot.metadata.json schema so the existing Jais
    # adapter/exporter, evaluator and TF-IDF scripts accept them unchanged.
    base = json.loads((args.frozen.parent / "train_pilot.metadata.json").read_text(encoding="utf-8"))
    identity = {"expansion_version": EXPANSION_VERSION, "nested_in": frozen["split_version"],
                "seed": frozen["seed"], "train_per_label": TARGET_PER_LABEL,
                "labels_sha256": frozen["labels_sha256"],
                "source_manifest_sha256": frozen["source_manifest_sha256"],
                "train_scaling_2000_sha256": fs.sha256_bytes(full_bytes),
                "train_scaling_new_1500_sha256": fs.sha256_bytes(new_bytes)}
    split_hash = fs.sha256_bytes(fs.canonical(identity))
    split_version = f"{EXPANSION_VERSION}-{split_hash[:12]}"
    provenance = {
        "issue": "EEC-24 / W2-02",
        "selection": {"method": "freeze_splits.sample_balanced rerun with train_per_label=100",
                      "seed": frozen["seed"], "freeze_version": fs.FREEZE_VERSION,
                      "freeze_code_sha256": fs.sha256_file(fs.__file__),
                      "expand_code_sha256": fs.sha256_file(__file__),
                      "dev_performance_used": False},
        "nested_in": {"split_version": frozen["split_version"],
                      "train_pilot_sha256": frozen["train_pilot_sha256"],
                      "contains_all_original_ids": True},
        "dev_pilot_unchanged": {"dev_pilot_sha256": frozen["dev_pilot_sha256"], "verified": True},
        "expansion_identity": identity,
    }

    def sidecar(payload, chosen, part):
        return {**base, "split_version": split_version, "split_sha256": split_hash,
                "manifest_sha256": fs.sha256_bytes(payload), "clips": len(chosen),
                "label_counts": {label: sum(r["label"] == label for r in chosen) for label in labels},
                "evaluation_label_sets": {"train": sorted({r["label"] for r in chosen}, key=labels.index)},
                "scaling_part": part, "w2_02_provenance": provenance}

    for name, payload, chosen, part in (
            ("train_scaling_2000", full_bytes, rows, "all_2000_nested"),
            ("train_scaling_new_1500", new_bytes, new_rows, "new_1500_only")):
        (manifest_out / f"{name}.metadata.json").write_bytes(fs.pretty(sidecar(payload, chosen, part)))
    print(f"Wrote {len(rows)} training clips ({len(new_rows)} new, 75 per label) to {manifest_out}")
    print(f"split_version {split_version}")
    print(f"train_scaling_2000.jsonl     sha256 {fs.sha256_bytes(full_bytes)}")
    print(f"train_scaling_new_1500.jsonl sha256 {fs.sha256_bytes(new_bytes)}")


if __name__ == "__main__":
    main()
