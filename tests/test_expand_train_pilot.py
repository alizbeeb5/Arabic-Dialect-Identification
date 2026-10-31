"""W2-02: the 2,000-clip training set is nested, balanced, training-only and leaves dev unchanged."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "data/manifests"


def ids(path):
    return [json.loads(line)["clip_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_committed_manifests_regenerate_byte_for_byte(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "src/data/expand_train_pilot.py"),
                    "--output-root", str(tmp_path)], check=True, cwd=ROOT)
    for name in ("train_scaling_2000.jsonl", "train_scaling_new_1500.jsonl",
                 "train_scaling_2000.metadata.json", "train_scaling_new_1500.metadata.json"):
        assert (tmp_path / "data/manifests" / name).read_bytes() == (M / name).read_bytes(), name


def test_nested_balanced_training_only():
    frozen = json.loads((M / "split_freeze.json").read_text(encoding="utf-8"))
    full, new = ids(M / "train_scaling_2000.jsonl"), ids(M / "train_scaling_new_1500.jsonl")
    assert len(full) == len(set(full)) == 2000 and len(new) == 1500
    assert set(frozen["train_pilot_ids"]) <= set(full)
    assert set(new) == set(full) - set(frozen["train_pilot_ids"])
    assert not set(full) & set(frozen["dev_pilot_ids"])
    rows = [json.loads(line) for line in (M / "train_scaling_2000.jsonl").read_text().splitlines() if line]
    assert all(r["split"] == "train" for r in rows)
    assert set(Counter(r["label"] for r in rows).values()) == {100}
