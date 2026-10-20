"""Acoustic encoder smoke run: acceptance checks, throughput and peak memory.

Runs a handful of manifest clips through the pinned encoder, writes the small
feature cache and records a run record under OUTPUT_ROOT/runs/<run_id>/.
Peak GPU memory is only measured on CUDA; CPU runs record process peak RSS.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acoustic_encoder as ae  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Max absolute difference allowed between two extractions of the same clip.
TOLERANCE = {"fp32": 1e-4, "bf16": 1e-1}
PACKAGES = ["torch", "torchaudio", "speechbrain", "huggingface_hub", "numpy", "soundfile"]


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def timed_extract(encoder, clips, batch_size):
    sync(encoder.device)
    started = time.perf_counter()
    result = encoder.extract(clips, batch_size=batch_size)
    sync(encoder.device)
    return result, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"))
    parser.add_argument("--cache-root", type=Path, default=os.environ.get("CACHE_ROOT"))
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    args = parser.parse_args(argv)
    if args.output_root is None:
        parser.error("--output-root or OUTPUT_ROOT is required")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / "runs" / f"acoustic_smoke_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = ae.FeatureConfig(precision=args.precision)
    tolerance = TOLERANCE[args.precision]

    t0 = time.perf_counter()
    encoder = ae.AcousticEncoder.from_pretrained(config, device=args.device, cache_dir=args.cache_root)
    sync(encoder.device)
    load_sec = time.perf_counter() - t0
    if encoder.device.type == "cuda":
        weights_bytes = torch.cuda.memory_allocated(encoder.device)

    rows = ae.read_manifest(args.manifest, args.limit)
    t0 = time.perf_counter()
    clips, hashes = [], []
    for row in rows:
        path = ae.resolve_audio_path(args.audio_root, row["audio_path"])
        hashes.append(ae.sha256_file(path))
        clips.append((row["clip_id"], ae.load_waveform(path, config)))
    audio_load_sec = time.perf_counter() - t0
    audio_sec = sum(wave.numel() for _, wave in clips) / config.sample_rate_hz
    durations = [wave.numel() / config.sample_rate_hz for _, wave in clips]

    # Warm-up batch (kernel selection, allocator) is excluded from timing.
    encoder.extract(clips[: args.batch_size], batch_size=args.batch_size)
    if encoder.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(encoder.device)
    first, first_sec = timed_extract(encoder, clips, args.batch_size)
    if encoder.device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(encoder.device)
        peak_reserved = torch.cuda.max_memory_reserved(encoder.device)
    second, second_sec = timed_extract(encoder, clips, args.batch_size)
    first.audio_sha256 = hashes

    # Padding invariance: the shortest clip alone versus batched with the longest.
    short, long_ = int(np.argmin(durations)), int(np.argmax(durations))
    alone = encoder.extract([clips[short]]).features[0]
    paired = encoder.extract([clips[short], clips[long_]], batch_size=2).features[0]

    cache = ae.FeatureCache(args.output_root, encoder.provenance())
    written = cache.write(first)
    reloaded = cache.load(list(reversed(first.clip_ids)))

    expected_shape = [len(clips), len(encoder.layers), encoder.output_dim]
    repeat_diff = float(np.abs(first.features - second.features).max())
    padding_diff = float(np.abs(alone - paired).max())
    cache_diff = float(np.abs(reloaded.features[::-1] - first.features).max())
    checks = {
        "finite": bool(np.isfinite(first.features).all()),
        "expected_shape": list(first.features.shape) == expected_shape,
        "clip_ids_preserved": first.clip_ids == [row["clip_id"] for row in rows]
        and reloaded.clip_ids == list(reversed(first.clip_ids)),
        "repeat_within_tolerance": repeat_diff <= tolerance,
        "padding_within_tolerance": padding_diff <= tolerance,
        # Newly written clips must read back exactly; clips cached by an earlier
        # run (e.g. another batch size) only need to agree within tolerance.
        "cache_consistent": cache_diff == 0.0 if written == len(clips) else cache_diff <= tolerance,
    }

    metrics = {
        "clips": len(clips), "audio_seconds": audio_sec,
        "clip_duration_sec": {"min": min(durations), "max": max(durations)},
        "batch_size": args.batch_size, "precision": args.precision,
        "feature_shape": list(first.features.shape), "layer_names": first.layer_names,
        "encoder_frames": {"min": int(first.n_frames.min()), "max": int(first.n_frames.max())},
        "model_load_sec": load_sec, "audio_load_sec": audio_load_sec,
        "extraction_sec": [first_sec, second_sec],
        "clips_per_sec": len(clips) / first_sec, "audio_sec_per_sec": audio_sec / first_sec,
        "tolerance_abs": tolerance, "repeat_max_abs_diff": repeat_diff,
        "padding_max_abs_diff": padding_diff, "cache_roundtrip_max_abs_diff": cache_diff,
        "cache_clips_written": written,
        "checks": checks, "process_peak_rss_bytes": peak_rss_bytes(),
        "peak_gpu_memory_allocated_bytes": None, "peak_gpu_memory_reserved_bytes": None,
    }
    hardware = {"device": str(encoder.device), "system": platform.system(),
                "architecture": platform.machine(), "processor": platform.processor() or None}
    if encoder.device.type == "cuda":
        metrics.update(peak_gpu_memory_allocated_bytes=peak_allocated,
                       peak_gpu_memory_reserved_bytes=peak_reserved,
                       gpu_weights_allocated_bytes=weights_bytes)
        hardware.update(gpu=torch.cuda.get_device_name(encoder.device),
                        gpu_total_memory_bytes=torch.cuda.get_device_properties(encoder.device).total_memory,
                        cuda=torch.version.cuda)

    finished = datetime.now(timezone.utc)
    run_config = {"feature_config": config.to_dict(), "manifest": str(args.manifest),
                  "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "limit": args.limit, "batch_size": args.batch_size}
    record = {
        "run_id": run_id, "experiment": "acoustic_encoder_smoke",
        "status": "completed" if all(checks.values()) else "failed_checks",
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "config": run_config,
        "config_sha256": hashlib.sha256(ae.canonical_json(run_config)).hexdigest(),
        "labels_sha256": hashlib.sha256((ROOT / "configs/labels.json").read_bytes()).hexdigest(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "hardware": hardware,
        "environment": {"python": platform.python_version(), "packages": package_versions()},
        "metrics": metrics, "encoder": encoder.provenance(), "feature_cache": str(cache.root),
        "output_dir": str(run_dir),
    }
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("feature_shape", "clips_per_sec", "audio_sec_per_sec",
                                               "repeat_max_abs_diff", "padding_max_abs_diff",
                                               "peak_gpu_memory_allocated_bytes", "checks")}, indent=2))
    print(f"Run record: {run_dir / 'run_record.json'}\nFeature cache: {cache.root}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
