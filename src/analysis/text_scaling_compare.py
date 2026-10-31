#!/usr/bin/env python3
"""W2-02 (EEC-24): compare the 500- and 2,000-clip text models on the frozen 200 dev IDs.

Reads only saved run directories written by src/models/transcript_tfidf.py.
Re-scores both dev prediction files with the shared evaluator, then reports
accuracy, macro-F1, per-dialect recall/F1, training fit, paired correctness
(with an exact two-sided McNemar test) and prediction sources.

Usage:
  python src/analysis/text_scaling_compare.py \
    --train-500 RUN --dev-500 RUN --train-2000 RUN --dev-2000 RUN --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation import evaluate as ev  # noqa: E402

DEV = ROOT / "data/manifests/dev_pilot.jsonl"
BASELINE = {"correct": 50, "macro_f1": 0.23642323859715164}


def load(run):
    run = Path(run)
    return (json.loads((run / "metrics.json").read_text()),
            json.loads((run / "run_record.json").read_text()),
            {p["clip_id"]: p for p in ev.read_jsonl(run / "predictions.jsonl")})


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("train-500", "dev-500", "train-2000", "dev-2000"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    labels = ev.load_labels(ROOT / "configs/labels.json")
    references = ev.validate_references(ev.read_jsonl(DEV), labels, "dev")
    result = {"dev_ids": len(references), "arms": {}}
    preds = {}
    for arm in ("500", "2000"):
        train_m, train_r, _ = load(getattr(args, f"train_{arm}"))
        dev_m, dev_r, dev_p = load(getattr(args, f"dev_{arm}"))
        if set(dev_p) != set(references):
            raise SystemExit(f"{arm}: dev predictions do not cover exactly the 200 frozen IDs")
        rescored = ev.compute_metrics(ev.join_predictions(
            references, ev.validate_prediction_fields(list(dev_p.values()), labels),
            allow_incomplete=False), labels)
        if rescored["correct"] != dev_m["correct"]:
            raise SystemExit(f"{arm}: saved dev metrics disagree with re-scoring")
        if dev_r["model_revision"] != train_r["model_revision"]:
            raise SystemExit(f"{arm}: dev run used a different checkpoint than its training run")
        diag = json.loads((Path(getattr(args, f"train_{arm}")) / "training_diagnostics.json").read_text())
        preds[arm] = dev_p
        result["arms"][arm] = {
            "checkpoint": train_r["model_revision"],
            "training_expected": diag["expected_training_clips"],
            "training_retained": diag["retained_training_clips"],
            "training_excluded": len(diag["excluded_training_ids"]),
            "retained_per_class_min_max": [min(diag["retained_training_class_counts"].values()),
                                           max(diag["retained_training_class_counts"].values())],
            "feature_counts": diag["feature_counts"],
            "fit_seconds": diag["fit_seconds"],
            "train_accuracy_all_ids": train_m["accuracy"],
            "dev_correct": rescored["correct"], "dev_accuracy": rescored["accuracy"],
            "dev_macro_f1": rescored["macro_f1"],
            "dev_prediction_sources": dev_r["prediction_sources"],
            "per_dialect": {c: {"recall": rescored["per_dialect"][c]["recall"],
                                "f1": rescored["per_dialect"][c]["f1"]} for c in labels},
            "dev_predictions_sha256": dev_m["predictions_sha256"],
        }

    base = result["arms"]["500"]
    result["baseline_500_matches_week1"] = (base["dev_correct"] == BASELINE["correct"]
                                            and abs(base["dev_macro_f1"] - BASELINE["macro_f1"]) < 1e-12)
    pairs = Counter()
    flips = []
    for clip_id, ref in references.items():
        a = preds["500"][clip_id]["predicted_label"] == ref["label"]
        b = preds["2000"][clip_id]["predicted_label"] == ref["label"]
        pairs[("both_correct", "only_500_correct", "only_2000_correct", "both_wrong")
              [0 if a and b else 1 if a else 2 if b else 3]] += 1
        if a != b:
            flips.append({"clip_id": clip_id, "label": ref["label"],
                          "pred_500": preds["500"][clip_id]["predicted_label"],
                          "pred_2000": preds["2000"][clip_id]["predicted_label"]})
    result["paired"] = dict(pairs)
    result["mcnemar_exact_p"] = mcnemar_exact(pairs["only_500_correct"], pairs["only_2000_correct"])
    result["flips"] = flips
    result["delta"] = {
        "dev_correct": result["arms"]["2000"]["dev_correct"] - base["dev_correct"],
        "dev_accuracy_points": 100 * (result["arms"]["2000"]["dev_accuracy"] - base["dev_accuracy"]),
        "dev_macro_f1": result["arms"]["2000"]["dev_macro_f1"] - base["dev_macro_f1"],
    }
    result["per_dialect_recall_change"] = {
        c: result["arms"]["2000"]["per_dialect"][c]["recall"] - base["per_dialect"][c]["recall"]
        for c in labels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'':>6} {'retained':>9} {'train acc':>9} {'dev correct':>12} {'dev acc':>8} {'macro-F1':>9}")
    for arm, r in result["arms"].items():
        print(f"{arm:>6} {r['training_retained']:>9} {r['train_accuracy_all_ids']:>9.3f} "
              f"{r['dev_correct']:>8}/200 {r['dev_accuracy']:>8.3f} {r['dev_macro_f1']:>9.4f}")
    print("Baseline reproduces Week 1 (50/200, same macro-F1):", result["baseline_500_matches_week1"])
    print("Paired:", result["paired"], f"| exact McNemar p = {result['mcnemar_exact_p']:.4f}")
    print("Recall change per dialect (2000 - 500):",
          {c: round(v, 1) for c, v in result["per_dialect_recall_change"].items()})
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
