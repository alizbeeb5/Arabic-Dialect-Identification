#!/usr/bin/env python3
"""Reproduce A predictions from a saved head and verified frozen-encoder features.

No fitting or checkpoint selection. Accepts any labeled manifest obeying the
shared contract, including separately frozen Casablanca diagnostic manifests.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as clf
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae


def main(argv=None, load_encoder=ae.AcousticEncoder.from_pretrained):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=clf.ROOT / "configs/labels.json")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    labels = ev.load_labels(args.labels)
    rows, split, meta = base.load_split(args.manifest, args.metadata, args.labels, labels)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["labels"] != labels or saved["config"].get("experiment", {}).get("architecture") != "A":
        raise ValueError("Expected architecture A with the canonical label order")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    encoder = load_encoder(clf.feature_config(saved["config"]["encoder"]),
                           device=args.device, cache_dir=args.cache_root)
    frozen = clf.frozen_summary(encoder)
    before = clf.state_sha256(encoder)
    provenance = encoder.provenance()
    digest = base.sha256_text(base.canonical_json(provenance))
    if digest != saved["feature_provenance_sha256"]:
        raise ValueError("Prediction encoder/features differ from the checkpoint provenance")
    cache = ae.FeatureCache(args.output_root, provenance)
    _, failures = clf.prepare_features(encoder, cache, base.without_labels(rows), args.audio_root,
                                       saved["config"]["encoder"], split)
    if before != clf.state_sha256(encoder):
        raise RuntimeError("Encoder state changed during prediction")
    del encoder
    failed = {f["clip_id"] for f in failures}
    ids = [r["clip_id"] for r in rows if r["clip_id"] not in failed]
    position = provenance["layer_names"].index(saved["layer_name"])
    head = clf.build_head(saved["feature_dim"], len(labels), seed=0)
    head.load_state_dict(saved["state_dict"], strict=True)
    logits, predicted = clf.predict(head, clf.load_layer(cache, ids, position))
    revision = "arabest-rq-frozen-linear-A:" + ev.sha256_file(args.checkpoint)[:16]
    predictions = {cid: {"clip_id": cid, "predicted_label": labels[int(p)],
                         "model_revision": revision, "status": "ok", "scores": row.tolist(),
                         "score_labels": labels, "score_type": "logits"}
                   for cid, p, row in zip(ids, predicted, logits)}
    for failure in failures:
        predictions[failure["clip_id"]] = {"clip_id": failure["clip_id"], "predicted_label": None,
            "model_revision": revision, "status": failure["status"], "error_code": failure["error_code"]}
    path = args.output_dir / "predictions.jsonl"
    base.write_jsonl(path, [predictions[r["clip_id"]] for r in rows])
    base.write_jsonl(args.output_dir / "failures.jsonl", failures)
    command = [sys.executable, str(clf.ROOT / "src/evaluation/evaluate.py"),
                   "--manifest", str(args.manifest), "--metadata", str(args.metadata),
                   "--predictions", str(path), "--labels", str(args.labels), "--split", split,
                   "--model-name", "arabest-rq-frozen-linear-A", "--output-dir",
                   str(args.output_dir / "evaluation")]
    completed = subprocess.run(command, cwd=clf.ROOT, text=True, capture_output=True)
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(f"Shared evaluator failed: {completed.stderr.strip()}")
    base.write_json(args.output_dir / "prediction_record.json", {
        "code": base.git_state(), "checkpoint_sha256": ev.sha256_file(args.checkpoint),
        "manifest_sha256": meta["manifest_sha256"], "metadata_sha256": ev.sha256_file(args.metadata),
        "feature_provenance_sha256": digest, "encoder_frozen": frozen,
        "predictions_sha256": ev.sha256_file(path), "elapsed_sec": time.perf_counter() - started,
        "expected": len(rows), "successful": len(ids), "failed": len(failures), "fit_performed": False})
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
