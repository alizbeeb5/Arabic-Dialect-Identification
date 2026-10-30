#!/usr/bin/env python3
"""Small CPU check that frozen Casablanca clips survive the acoustic preprocessing path.

Runs acoustic_encoder.load_waveform (mean downmix, sinc resample to 16 kHz, full clip)
on the longest clips plus a seeded random sample. No model is loaded and nothing is scored.
"""

import argparse
import hashlib
import json
import platform
import random
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
import acoustic_encoder as ae  # noqa: E402
import prepare as adi  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/casablanca_validation.jsonl")
    parser.add_argument("--longest", type=int, default=10)
    parser.add_argument("--random", type=int, default=40)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports/casablanca_validation_audit/preprocessing_check.json")
    return parser.parse_args(argv)


def select(rows, longest, count, seed):
    ok = [r for r in rows if r["preparation_status"] == "ok"]
    by_length = sorted(ok, key=lambda r: (-r["frame_count"] / r["sample_rate_hz"], r["clip_id"]))
    chosen = {r["clip_id"]: "longest" for r in by_length[:longest]}
    rest = sorted(r["clip_id"] for r in ok if r["clip_id"] not in chosen)
    for clip_id in random.Random(seed).sample(rest, min(count, len(rest))):
        chosen[clip_id] = "random"
    index = {r["clip_id"]: r for r in ok}
    return [(index[c], reason) for c, reason in chosen.items()]


def main(argv=None):
    import torch
    import torchaudio

    args = parse_args(argv)
    config = ae.FeatureConfig()
    metadata = json.loads(args.manifest.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != adi.sha256_file(args.manifest):
        raise ValueError("Manifest does not match its metadata sidecar")
    audio_root = args.data_root.expanduser().resolve() / metadata["audio_root"]
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    results, failures = [], []
    started = time.monotonic()
    for row, reason in select(rows, args.longest, args.random, args.seed):
        path = ae.resolve_audio_path(audio_root, row["audio_path"])
        clip_started = time.perf_counter()
        try:
            if adi.sha256_file(path) != row["audio_sha256"]:
                raise ValueError("Extracted audio differs from manifest hash")
            wave = ae.load_waveform(path, config)
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"clip_id": row["clip_id"], "reason": str(error)})
            continue
        expected = row["frame_count"] * config.sample_rate_hz / row["sample_rate_hz"]
        results.append({"clip_id": row["clip_id"], "label": row["label"], "selected_as": reason,
                        "input_sample_rate_hz": row["sample_rate_hz"], "input_channels": row["channel_count"],
                        "input_frames": row["frame_count"], "duration_sec": row["duration_sec"],
                        "output_samples": int(wave.numel()), "expected_output_samples": expected,
                        "length_ok": abs(wave.numel() - expected) <= 1,
                        "finite": bool(torch.isfinite(wave).all()),
                        "output_peak_abs": float(wave.abs().max()),
                        "output_sha256": hashlib.sha256(wave.numpy().astype("<f4").tobytes()).hexdigest(),
                        "seconds": round(time.perf_counter() - clip_started, 4)})
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"status": "passed" if not failures and all(r["length_ok"] and r["finite"] for r in results)
              else "failed",
              "checked": len(results), "failures": failures, "seed": args.seed,
              "feature_config": config.to_dict(), "feature_config_sha256": config.sha256(),
              "manifest_sha256": metadata["manifest_sha256"], "split_version": metadata["split_version"],
              "longest_checked": max(results, key=lambda r: r["duration_sec"], default=None),
              "wall_seconds": round(time.monotonic() - started, 2),
              # ru_maxrss is bytes on macOS, KiB on Linux.
              "peak_rss_mib": round(peak_rss / (2**20 if sys.platform == "darwin" else 2**10), 1),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "torchaudio": torchaudio.__version__},
              "hardware": {"platform": platform.platform(), "machine": platform.machine()},
              "note": "Preprocessing only; no encoder loaded. Model memory/throughput is benchmarked at scoring time.",
              "clips": results}
    adi.atomic_write(args.output, adi.json_bytes(report))
    print(f"{report['status']}: {len(results)} clips checked, {len(failures)} failures -> {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
