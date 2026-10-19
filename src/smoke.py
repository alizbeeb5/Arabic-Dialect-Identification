import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]


def git_output(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started = datetime.now(timezone.utc)

# Read the experiment settings and label mapping.
config_path = ROOT / "configs/example.yaml"
config = yaml.safe_load(config_path.read_text())

labels_path = ROOT / config["paths"]["labels"]
labels = json.loads(labels_path.read_text())["label_order"]

if len(labels) != 20 or len(set(labels)) != 20:
    raise ValueError("Expected 20 unique dialect labels.")

if config["data"]["dataset"] != "synthetic":
    raise ValueError("This script only runs the synthetic smoke check.")

# Create four synthetic examples with exactly one wrong prediction.
rng = np.random.default_rng(config["experiment"]["seed"])
true_ids = rng.integers(0, len(labels), size=4)
predicted_ids = true_ids.copy()
predicted_ids[-1] = (true_ids[-1] + 1) % len(labels)

accuracy = float(accuracy_score(true_ids, predicted_ids))

# Give every execution its own output directory.
run_id = started.strftime("%Y%m%dT%H%M%S%fZ")
output_dir = ROOT / config["paths"]["output_dir"] / run_id
output_dir.mkdir(parents=True, exist_ok=False)

finished = datetime.now(timezone.utc)

record = {
    "run_id": run_id,
    "experiment": config["experiment"]["name"],
    "status": "completed",
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_dirty": bool(git_output("status", "--porcelain")),
    "config": config,
    "config_sha256": sha256(config_path),
    "labels_sha256": sha256(labels_path),
    "command": shlex.join([sys.executable, *sys.argv]),
    "started_at": started.isoformat(),
    "finished_at": finished.isoformat(),
    "duration_seconds": (finished - started).total_seconds(),
    "hardware": {
        "device": "cpu",
        "system": platform.system(),
        "architecture": platform.machine(),
    },
    "environment": {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["PyYAML", "numpy", "scikit-learn"]
        },
    },
    "metrics": {
        "synthetic_examples": 4,
        "synthetic_accuracy": accuracy,
    },
    "output_dir": str(output_dir.relative_to(ROOT)),
}

record_path = output_dir / "run_record.json"
record_path.write_text(json.dumps(record, indent=2) + "\n")

print(f"Synthetic accuracy: {accuracy:.2f}")
print(f"Run record: {record_path.relative_to(ROOT)}")