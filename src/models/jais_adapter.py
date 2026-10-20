"""Pinned Jais-ASR inference, hard-label predictions and per-clip resume receipts.

Model code receives waveforms/rates/IDs only. No gold dialect, file path or
source metadata is sent to the processor or generation prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "IFM/Jais-ASR"
MODEL_REVISION = "1323995fa189c479369a3929be0e95039eba5cba"
ADAPTER_VERSION = "jais-adapter-v1"
JAIS_TO_ADI20 = {
    "msa": "MSA", "BH": "BAH", "TN": "TUN", "DZ": "ALG",
    "EG": "EGY", "IQ": "IRA", "JO": "JOR", "SA": "KSA",
    "KW": "KUW", "LB": "LEB", "LY": "LIB", "MR": "MAU",
    "MA": "MOR", "OM": "OMA", "PS": "PAL", "QA": "QAT",
    "SD": "SUD", "SY": "SYR", "AE": "UAE", "YE": "YEM",
}
PACKAGES = ("torch", "transformers", "accelerate", "numpy", "soundfile", "soxr")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(json_bytes(value) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def failure(clip_id, revision, status, code, error=None):
    record = {
        "prediction": {"clip_id": clip_id, "predicted_label": None,
                       "model_revision": revision, "status": status,
                       "error_code": code},
        "raw_output": None, "transcript": None, "native_dialect": None,
        "refused": None, "truncated": None,
    }
    if error is not None:
        record.update(error_type=type(error).__name__, error_message=str(error))
    return record


def parsed_record(clip_id, revision, raw_text, item):
    """A dialect tag is a hard label. Truncation remains a separate text flag."""
    native = item.language
    refused = item.refused
    truncated = getattr(item, "truncated", None)
    if not isinstance(item.text, str) or not isinstance(raw_text, str):
        raise ValueError("Transcript and raw output must be strings")
    if native is not None and not isinstance(native, str):
        raise ValueError("Native dialect must be a string or null")
    if type(refused) is not bool or (truncated is not None and type(truncated) is not bool):
        raise ValueError("Invalid refusal/truncation flag")
    label = JAIS_TO_ADI20.get(native)
    code = "refused" if refused else "unrecognized_dialect"
    record = failure(clip_id, revision, "inference_error", code)
    if label is not None and not refused:
        record["prediction"] = {"clip_id": clip_id, "predicted_label": label,
                                "model_revision": revision, "status": "ok"}
    record.update(raw_output=raw_text, transcript=item.text,
                  native_dialect=native, refused=refused, truncated=truncated)
    return record


def validate_record(record, clip_id, revision):
    if not isinstance(record, dict):
        raise ValueError("Expected an identified output record")
    prediction = record.get("prediction", {})
    if not isinstance(prediction, dict) or not {
        "clip_id", "predicted_label", "model_revision", "status"
    }.issubset(prediction):
        raise ValueError("Missing prediction contract fields")
    if prediction.get("clip_id") != clip_id or prediction.get("model_revision") != revision:
        raise ValueError("Output clip ID or checkpoint revision does not match")
    if any(key in prediction for key in ("scores", "score_labels", "score_type", "confidence")):
        raise ValueError("Jais dialect tags do not provide class-score vectors")
    status = prediction.get("status")
    if status == "ok":
        if prediction.get("predicted_label") not in JAIS_TO_ADI20.values():
            raise ValueError("Invalid canonical dialect label")
        if record.get("refused") is not False:
            raise ValueError("A refused result cannot be an OK prediction")
        if JAIS_TO_ADI20.get(record.get("native_dialect")) != prediction["predicted_label"]:
            raise ValueError("Canonical prediction disagrees with native tag")
    elif status not in {"audio_error", "inference_error"} or prediction.get("predicted_label") is not None:
        raise ValueError("Invalid failure prediction")
    for key in ("raw_output", "transcript", "native_dialect", "refused", "truncated"):
        if key not in record:
            raise ValueError(f"Missing output field: {key}")
    for key in ("raw_output", "transcript", "native_dialect"):
        if record[key] is not None and not isinstance(record[key], str):
            raise ValueError(f"Invalid output type: {key}")
    for key in ("refused", "truncated"):
        if record[key] is not None and type(record[key]) is not bool:
            raise ValueError(f"Invalid output flag: {key}")


def decode_audio(payload):
    import io
    import numpy as np
    import soundfile as sf

    waveform, rate = sf.read(io.BytesIO(payload), dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if (waveform.ndim != 1 or waveform.size == 0 or rate <= 0
            or not np.isfinite(waveform).all()):
        raise ValueError("Empty or invalid decoded audio")
    return waveform, rate


class JaisAdapter:
    """Use from_pretrained(), or reuse an already loaded Colab model/processor."""

    def __init__(self, model, processor, *, revision=MODEL_REVISION,
                 model_id=MODEL_ID, max_new_tokens=128):
        import torch

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use an immutable 40-character model revision")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.model, self.processor = model.eval(), processor
        self.revision, self.max_new_tokens = revision, max_new_tokens
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError("This adapter expects a single-device model")
        processor.tokenizer.padding_side = "left"
        self.prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio"}]}],
            tokenize=False, add_generation_prompt=True,
        )
        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.identity = {
            "model_id": model_id, "model_revision": revision,
            "dtype": str(self.dtype), "device": str(self.device),
            "prompt": self.prompt, "padding_side": "left",
            "mono_conversion": "channel_mean", "audio_dtype": "float32",
            "do_sample": False, "max_new_tokens": max_new_tokens,
            "generation_defaults": model.generation_config.to_dict(),
            "label_mapping": dict(JAIS_TO_ADI20), "packages": packages,
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda" else None),
        }

    @classmethod
    def from_pretrained(cls, *, revision=MODEL_REVISION, dtype="float16",
                        max_new_tokens=128, cache_dir=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use an immutable 40-character model revision")
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported floating-point dtype")
        options = dict(revision=revision, trust_remote_code=True, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(MODEL_ID, **options)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, **options, dtype=getattr(torch, dtype), device_map="auto",
        )
        return cls(model, processor, revision=revision, max_new_tokens=max_new_tokens)

    def predict(self, clip_ids, waveforms, sample_rates):
        import torch

        if not (len(clip_ids) == len(waveforms) == len(sample_rates)):
            raise ValueError("Clip IDs, waveforms and sample rates must align")
        if not clip_ids:
            return []
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("Duplicate clip IDs")
        inputs = self.processor(
            text=[self.prompt] * len(clip_ids), audio=waveforms,
            sampling_rate=sample_rates, padding=True, return_tensors="pt",
        ).to(self.device)
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(dtype=self.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
            )
        length = inputs["input_ids"].shape[1]
        if generated.shape[0] != len(clip_ids):
            raise RuntimeError("Unexpected generation batch size")
        tails = generated[:, length:]
        token_ids = tails.detach().cpu().tolist()
        # Capture raw output before parsing so parser failures retain evidence.
        raw = self.processor.tokenizer.batch_decode(tails, skip_special_tokens=False)
        if len(raw) != len(clip_ids):
            raise RuntimeError("Unexpected raw-output count")
        try:
            parsed = self.processor.post_process_transcription(generated, input_lengths=length)
            if len(parsed) != len(clip_ids):
                raise ValueError("Unexpected parsed-output count")
        except Exception as error:
            records = []
            for clip_id, text, tokens in zip(clip_ids, raw, token_ids, strict=True):
                record = failure(clip_id, self.revision, "inference_error", "parse_failed", error)
                record.update(raw_output=text, raw_token_ids=tokens)
                records.append(record)
            return records
        records = []
        for clip_id, text, tokens, item in zip(clip_ids, raw, token_ids, parsed, strict=True):
            try:
                record = parsed_record(clip_id, self.revision, text, item)
            except Exception as error:
                record = failure(clip_id, self.revision, "inference_error", "parse_failed", error)
                record["raw_output"] = text
            record["raw_token_ids"] = tokens
            records.append(record)
        return records


def safe_audio_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio_path must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("audio_path must stay relative to audio_root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio path escapes audio_root")
    return path


def run_cached_jais(clips, audio_root, cache_root, adapter, *, batch_size=2,
                    decoder=decode_audio, progress=None):
    """Return one record per input, in input order; retry transient failures.

    Successful outputs, refusals and unrecognized tags have resume receipts.
    File-read, decode, generation and parse errors are returned but not cached.
    Corrupt/inconsistent receipts reject the run rather than alter its results.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = Path(audio_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"audio_root is not a directory: {root}")
    cache = Path(cache_root).expanduser().resolve()
    ids, paths = [], []
    for row in clips:
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("clip_id must be a nonempty string")
        ids.append(clip_id)
        paths.append(safe_audio_path(root, row.get("audio_path")))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate clip IDs")
    config = {
        "adapter_version": ADAPTER_VERSION, "adapter_code_sha256": sha256_file(__file__),
        "backend": adapter.identity, "batch_size": batch_size,
    }
    # Snapshot mutable defaults before inference; never retain aliases to them.
    config = json.loads(json_bytes(config))
    results, pending = {}, []
    stats = {"expected": len(ids), "cache_hits": 0, "model_batches": 0, "new_outputs": 0}

    def notify():
        if progress is not None:
            progress({**stats, "completed": len(results)})

    def flush():
        if not pending:
            return
        stats["model_batches"] += 1
        try:
            records = adapter.predict(
                [p["clip_id"] for p in pending], [p["waveform"] for p in pending],
                [p["rate"] for p in pending],
            )
            if len(records) != len(pending):
                raise ValueError("Backend returned an unexpected number of records")
            for entry, record in zip(pending, records, strict=True):
                validate_record(record, entry["clip_id"], adapter.revision)
        except Exception as error:
            for entry in pending:
                results[entry["clip_id"]] = failure(
                    entry["clip_id"], adapter.revision, "inference_error", "batch_failed", error,
                )
        else:
            for entry, record in zip(pending, records, strict=True):
                prediction = record["prediction"]
                if prediction["status"] == "ok" or prediction.get("error_code") in {
                    "refused", "unrecognized_dialect",
                }:
                    atomic_json(entry["cache_path"], {
                        "identity": entry["identity"], "record": record,
                        "record_sha256": hashlib.sha256(json_bytes(record)).hexdigest(),
                    })
                results[entry["clip_id"]] = record
                stats["new_outputs"] += 1
        pending.clear()
        notify()

    for clip_id, path in zip(ids, paths, strict=True):
        try:
            payload = path.read_bytes()
        except (OSError, ValueError) as error:
            results[clip_id] = failure(clip_id, adapter.revision, "audio_error", "read_failed", error)
            notify()
            continue
        identity = {"clip_id": clip_id, "audio_sha256": hashlib.sha256(payload).hexdigest(),
                    "config": config}
        key = hashlib.sha256(json_bytes(identity)).hexdigest()
        cache_path = cache / (key + ".json")
        if cache_path.exists():
            try:
                receipt = read_json(cache_path)
                record = receipt["record"]
                if (receipt["identity"] != identity or receipt["record_sha256"]
                        != hashlib.sha256(json_bytes(record)).hexdigest()):
                    raise ValueError("Receipt identity/content mismatch")
                validate_record(record, clip_id, adapter.revision)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid cache receipt {cache_path}: {error}") from error
            results[clip_id] = record
            stats["cache_hits"] += 1
            notify()
            continue
        try:
            waveform, rate = decoder(payload)
        except Exception as error:
            results[clip_id] = failure(clip_id, adapter.revision, "audio_error", "decode_failed", error)
            notify()
            continue
        pending.append(dict(clip_id=clip_id, waveform=waveform, rate=rate,
                            identity=identity, cache_path=cache_path))
        if len(pending) >= batch_size:
            flush()
    flush()
    ordered = [results[clip_id] for clip_id in ids]
    stats.update(successful=sum(r["prediction"]["status"] == "ok" for r in ordered),
                 failed=sum(r["prediction"]["status"] != "ok" for r in ordered))
    return ordered, stats


def write_outputs(output_dir, records, run_record):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("raw_outputs.jsonl", records),
                       ("predictions.jsonl", [r["prediction"] for r in records])):
        path = output / name
        if path.exists():
            raise ValueError(f"Refusing to overwrite run output: {path}")
        path.write_bytes(b"".join(json_bytes(row) + b"\n" for row in rows))
    run_record["predictions_sha256"] = sha256_file(output / "predictions.jsonl")
    run_record["raw_outputs_sha256"] = sha256_file(output / "raw_outputs.jsonl")
    atomic_json(output / "run_record.json", run_record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, help="Override the metadata storage root")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new output directory; reuse --cache-root to resume")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and token limit must be positive")
    try:
        metadata = read_json(args.metadata)
        if sha256_file(args.manifest) != metadata["manifest_sha256"]:
            raise ValueError("Manifest hash does not match its metadata")
        with args.manifest.open(encoding="utf-8") as file:
            clips = [json.loads(line) for line in file if line.strip()]
        if not clips:
            raise ValueError("Manifest has no clips")
        root = args.audio_root or Path(metadata["audio_root"])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "run_id": args.output_dir.name, "experiment": "jais_batch_inference",
            "status": "running", "git_commit": None, "git_dirty": None,
            "adapter_code_sha256": sha256_file(__file__), "adapter_version": ADAPTER_VERSION,
            "model_id": MODEL_ID, "model_revision": args.revision,
            "dataset_version": metadata.get("dataset_version"),
            "split_version": metadata.get("split_version"),
            "manifest_sha256": metadata["manifest_sha256"],
            "metadata_sha256": sha256_file(args.metadata),
            "labels_sha256": metadata.get("labels_sha256"),
            "command": ([sys.executable, *sys.argv] if argv is None
                        else [sys.executable, str(Path(__file__).resolve()), *argv]),
            "audio_root": str(root), "cache_root": str(args.cache_root),
            "output_dir": str(args.output_dir), "expected": len(clips),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "timing_scope": "model loading, audio/cache reads, preprocessing, generation and cache writes; excludes final export",
        }
        git_root = Path(__file__).resolve().parents[2]
        try:
            record["git_commit"] = subprocess.check_output(
                ["git", "-C", str(git_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip()
            record["git_dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(git_root), "status", "--porcelain"], stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            pass
        atomic_json(args.output_dir / "run_record.json", record)
        print(f"Processing {len(clips)} clips from {args.manifest}", flush=True)
        start = time.perf_counter()
        try:
            adapter = JaisAdapter.from_pretrained(
                revision=args.revision, dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                cache_dir=str(args.model_cache) if args.model_cache else None,
            )
            import torch
            if adapter.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
                torch.cuda.reset_peak_memory_stats(adapter.device)
            record["inference_config"] = adapter.identity

            def progress(counts):
                record["counts"] = counts
                print(f"Completed {counts['completed']}/{len(clips)}; "
                      f"cached={counts['cache_hits']}; batches={counts['model_batches']}", flush=True)

            records, counts = run_cached_jais(
                clips, root, args.cache_root, adapter, batch_size=args.batch_size, progress=progress,
            )
            if adapter.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
                record["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(adapter.device) / 2**30
            record.update(counts=counts, wall_seconds=time.perf_counter() - start,
                          status="completed" if counts["failed"] == 0 else "completed_with_failures",
                          finished_at=datetime.now(timezone.utc).isoformat())
            write_outputs(args.output_dir, records, record)
            print(json.dumps(counts), flush=True)
            return 2 if counts["failed"] else 0
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error_type=type(error).__name__, error_message=str(error),
                          wall_seconds=time.perf_counter() - start,
                          finished_at=datetime.now(timezone.utc).isoformat())
            atomic_json(args.output_dir / "run_record.json", record)
            raise
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"Jais adapter error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
