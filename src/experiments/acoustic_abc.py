#!/usr/bin/env python3
"""One entry point for A/B/C training and saved-checkpoint prediction (EEC-23)."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from importlib.metadata import version

import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation import evaluate as ev
from src.experiments import acoustic_classifier as reference
from src.experiments import acoustic_A_predict
from src.experiments import mfcc_svm_baseline as base
from src.models import acoustic_encoder as ae
from src.models.acoustic_architectures import AcousticDialectModel

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "configs/labels.json"
CONFIGS = ROOT / "configs/acoustic"
# Must be set before the first CUDA BLAS operation for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_config(architecture):
    config = json.loads((CONFIGS / f"{architecture}.json").read_text())
    if architecture == "A":
        return reference.load_config(CONFIGS / "A.json")
    enc, model, opt = config["encoder"], config["model"], config["optimization"]
    if (config["experiment"]["architecture"] != architecture or enc["checkpoint"] != ae.REPO_ID
            or enc["revision"] != ae.REVISION or enc["precision"] != "fp32"
            or enc["max_duration_sec"] is not None or enc["frozen"] != (architecture == "B")):
        raise ValueError("B/C require the pinned full-clip FP32 encoder and declared freezing")
    if model != {"pooling": "masked_additive_attention", "attention_dim": 128,
                 "feature_dim": 848, "hidden_dim": 256, "dropout": 0.2, "num_labels": 20}:
        raise ValueError("B/C must share the approved attention + 256-unit GELU MLP")
    if opt["optimizer"] != "AdamW" or opt["loss"] != "cross_entropy" or opt["checkpoint"] != "last_epoch":
        raise ValueError("Only fixed AdamW/CE/last-epoch training is supported")
    for key in ("epochs", "effective_batch_size", "micro_batch_size"):
        if type(opt[key]) is not int or opt[key] < 1:
            raise ValueError(f"Invalid {key}")
    if opt["micro_batch_size"] != 1:
        raise ValueError("The declared full-clip memory policy uses one clip per forward")
    if min(opt["head_lr"], opt["encoder_lr"], opt["gradient_clip_norm"]) <= 0 or opt["weight_decay"] < 0:
        raise ValueError("Learning rates and clipping must be positive; weight decay nonnegative")
    return config


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def source_identity():
    files = [Path(__file__), Path(ae.__file__), ROOT / "src/models/acoustic_architectures.py"]
    return {**base.git_state(), "sha256": {str(p.relative_to(ROOT)): ev.sha256_file(p) for p in files}}


def parameter_sha256(module):
    """Parameter-only fingerprint: BN buffer changes alone do not prove fine-tuning."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_state(model):
    state = {"pooling": model.pooling.state_dict(), "head": model.head.state_dict()}
    if model.architecture == "C":
        state["encoder"] = model.encoder.model.state_dict()
    return state


def restore_model(model, state):
    model.pooling.load_state_dict(state["pooling"], strict=True)
    model.head.load_state_dict(state["head"], strict=True)
    if model.architecture == "C":
        model.encoder.model.load_state_dict(state["encoder"], strict=True)


def optimizer_for(model, config):
    o = config["optimization"]
    return torch.optim.AdamW(model.parameter_groups(o["encoder_lr"], o["head_lr"], o["weight_decay"]),
                             betas=(0.9, 0.999), eps=1e-8)


def load_audio(row, root, feature_config):
    # Audio-only boundary: labels, filenames and source metadata are never predictive features.
    row = base.without_labels([row])[0]
    path = ae.resolve_audio_path(root, row["audio_path"])
    base.verify_audio(path, row)
    return ae.load_waveform(path, feature_config)


def smallest_per_label(rows, labels, count):
    chosen = []
    for label in labels:
        candidates = sorted((r for r in rows if r["label"] == label),
                            key=lambda r: (r["duration_sec"], r["clip_id"]))
        if len(candidates) < count:
            raise ValueError(f"Not enough training examples for {label}")
        chosen.extend(candidates[:count])
    return chosen


def audit_audio(rows, root, config, split):
    ready, failures = [], []
    for i, row in enumerate(rows):
        try:
            if row.get("preparation_status") == "audio_error":
                raise ValueError("Manifest declares an audio preparation failure")
            load_audio(row, root, config)
            ready.append(row)
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"clip_id": row["clip_id"], "split": split, "status": "audio_error",
                             "error_code": "audio_validation_failed", "message": str(error)[:500]})
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            base.progress(f"[audio audit {split}] {i+1}/{len(rows)}, failures={len(failures)}")
    return ready, failures


def gradient_summary(model):
    def nonzero(params):
        return sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params)
    return {"encoder_parameters_with_nonzero_gradient": nonzero(model.encoder.parameters()),
            "pooling_parameters_with_nonzero_gradient": nonzero(model.pooling.parameters()),
            "head_parameters_with_nonzero_gradient": nonzero(model.head.parameters())}


def train_epoch(model, rows, labels, audio_root, optimizer, config, seed, epoch):
    """One exposure per retained original clip; exact mean loss for partial accumulations."""
    model.train()
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    effective = config["optimization"]["effective_batch_size"]
    total_loss = correct = updates = 0
    gradients = None
    for start in range(0, len(order), effective):
        group = [rows[i] for i in order[start:start + effective]]
        optimizer.zero_grad(set_to_none=True)
        for row in group:
            wave = load_audio(row, audio_root, model.encoder.config)
            logits = model([wave])
            target = torch.tensor([labels.index(row["label"])], device=logits.device)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {row['clip_id']}")
            (loss / len(group)).backward()
            total_loss += float(loss.detach())
            correct += int(logits.argmax(1).item() == target.item())
        if gradients is None:
            gradients = gradient_summary(model)
            if model.architecture == "B" and gradients["encoder_parameters_with_nonzero_gradient"]:
                raise RuntimeError("Frozen B encoder received gradients")
            if model.architecture == "C" and not gradients["encoder_parameters_with_nonzero_gradient"]:
                raise RuntimeError("C has no encoder gradients; fine-tuning is disconnected")
            if not gradients["pooling_parameters_with_nonzero_gradient"] or not gradients["head_parameters_with_nonzero_gradient"]:
                raise RuntimeError("Attention or classifier is disconnected")
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             config["optimization"]["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        if updates % 100 == 0:
            base.progress(f"[epoch {epoch}] {min(start+effective,len(rows))}/{len(rows)}, loss={total_loss/(start+len(group)):.4f}")
    if not rows:
        raise ValueError("No usable training clips")
    return {"loss": total_loss / len(rows), "accuracy": correct / len(rows), "clips": len(rows),
            "updates": updates, "last_gradient_norm": float(norm), "gradients": gradients}


@torch.no_grad()
def predict_rows(model, rows, audio_root, labels, revision):
    model.eval()
    predictions, failures = [], []
    for row in rows:
        try:
            if row.get("preparation_status") == "audio_error":
                raise ValueError("Manifest declares an audio preparation failure")
            wave = load_audio(row, audio_root, model.encoder.config)
        except (OSError, ValueError, RuntimeError) as error:
            status, error_code = "audio_error", "audio_validation_failed"
            logits = None
        else:
            try:
                logits = model([wave])[0]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite logits")
            except (RuntimeError, FloatingPointError) as error:
                status, error_code, logits = "inference_error", "encoder_or_head_failed", None
                if model.encoder.device.type == "cuda":
                    torch.cuda.empty_cache()
        if logits is None:
            failures.append({"clip_id": row["clip_id"], "status": status, "error_code": error_code})
            predictions.append({"clip_id": row["clip_id"], "predicted_label": None,
                                "status": status, "error_code": error_code, "model_revision": revision})
        else:
            scores = logits.float().cpu().tolist()
            predictions.append({"clip_id": row["clip_id"], "predicted_label": labels[max(range(len(labels)), key=scores.__getitem__)],
                                "scores": scores, "score_labels": labels, "score_type": "logits",
                                "status": "ok", "model_revision": revision})
    ev.validate_prediction_fields(predictions, labels)
    return predictions, failures


def evaluate_predictions(rows, predictions, labels, output, args):
    refs = ev.validate_references(rows, labels, "dev")
    metrics = ev.compute_metrics(ev.join_predictions(refs, ev.validate_prediction_fields(predictions, labels)), labels)
    # Use the canonical evaluator to also export its confusion-matrix CSV and provenance.
    command = [sys.executable, str(ROOT / "src/evaluation/evaluate.py"), "--manifest", str(args.dev_manifest),
               "--metadata", str(args.dev_manifest.with_suffix('.metadata.json')), "--labels", str(LABELS),
               "--predictions", str(output.parent / "predictions.jsonl"), "--split", "dev",
               "--model-name", f"acoustic-{args.architecture}", "--output-dir", str(output)]
    subprocess.run(command, check=True, cwd=ROOT)
    return metrics


def preflight(model, optimizer, train_rows, dev_rows, root, labels, output):
    """Measure full longest training forward/backward/AdamW step and dev inference."""
    checks = []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        row = max(rows, key=lambda r: r["duration_sec"])
        item = {"split": split, "clip_id": row["clip_id"], "duration_sec": row["duration_sec"]}
        tick = time.perf_counter()
        try:
            if model.encoder.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.encoder.device)
                torch.cuda.synchronize(model.encoder.device)
            wave = load_audio(row, root, model.encoder.config)
            model.train(split == "train")
            if split == "train":
                optimizer.zero_grad(set_to_none=True)
                logits = model([wave])
                F.cross_entropy(logits, torch.tensor([labels.index(row["label"])], device=logits.device)).backward()
                optimizer.step()  # Allocate AdamW moments, not just backward buffers.
                item["gradients"] = gradient_summary(model)
            else:
                with torch.no_grad():
                    logits = model([wave])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite preflight logits")
            item.update(status="ok", output_shape=list(logits.shape))
        except Exception as error:
            item.update(status="failed", error_type=type(error).__name__, message=str(error)[:500])
        if model.encoder.device.type == "cuda":
            torch.cuda.synchronize(model.encoder.device)
            item["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(model.encoder.device)
        item["seconds"] = time.perf_counter() - tick
        checks.append(item)
        base.write_json(output, {"passed": all(c["status"] == "ok" for c in checks), "checks": checks,
                                "policy": "full clips; one original clip per forward; no crop or windowing",
                                "frame_storage": "on-the-fly for B and C; no mean/frame cache",
                                "estimated_full_fp32_final_frame_cache_bytes": sum(
                                    math.ceil((r["duration_sec"] or 0) / 0.04) * model.encoder.architecture.d_model * 4
                                    for r in train_rows + dev_rows)})
        if item["status"] != "ok":
            raise RuntimeError(f"{split} longest-clip preflight failed; see {output}")


def overfit_gate(model, rows, labels, root, config, output):
    gate_rows = smallest_per_label(rows, labels, config["overfit_check"]["clips_per_label"])
    settings = copy.deepcopy(config)
    settings["optimization"]["effective_batch_size"] = len(gate_rows)
    optimizer = optimizer_for(model, settings)
    initial = None
    for step in range(config["overfit_check"]["epochs"]):
        result = train_epoch(model, gate_rows, labels, root, optimizer, settings, config["seed"], step)
        if initial is None:
            initial = result["loss"]
        predictions, failures = predict_rows(model, gate_rows, root, labels, "overfit-gate")
        accuracy = sum(p["predicted_label"] == r["label"] for p, r in zip(predictions, gate_rows)) / len(gate_rows)
        passed = not failures and accuracy >= config["overfit_check"]["min_accuracy"] and result["loss"] <= initial * config["overfit_check"]["max_loss_ratio"]
        record = {"passed": passed, "epochs": step+1, "initial_loss": initial, "final_loss": result["loss"],
                  "accuracy": accuracy, "clip_ids": [r["clip_id"] for r in gate_rows], "training_only": True}
        base.write_json(output, record)
        if passed:
            return record
    raise RuntimeError(f"Training-only overfit check failed; see {output}")


def run_bc(args, config, seed, output, load_encoder=ae.AcousticEncoder.from_pretrained):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"status": "starting", "architecture": args.architecture, "seed": seed, "source": source_identity(),
              "full_training_result": args.stage in ("select", "report"), "stage": args.stage}
    base.write_json(output / "run_record.json", record)
    try:
        labels = ev.load_labels(LABELS)
        train, train_split, train_meta = base.load_split(args.train_manifest, args.train_manifest.with_suffix('.metadata.json'), LABELS, labels)
        dev, dev_split, dev_meta = base.load_split(args.dev_manifest, args.dev_manifest.with_suffix('.metadata.json'), LABELS, labels)
        if (train_split, dev_split) != ("train", "dev") or {r["clip_id"] for r in train} & {r["clip_id"] for r in dev}:
            raise ValueError("Expected disjoint train and dev manifests")
        reference.validate_experiment_data(config, train, dev, train_meta, dev_meta)
        binding = {"train_manifest_sha256": ev.sha256_file(args.train_manifest),
                   "dev_manifest_sha256": ev.sha256_file(args.dev_manifest),
                   "train_metadata_sha256": ev.sha256_file(args.train_manifest.with_suffix('.metadata.json')),
                   "dev_metadata_sha256": ev.sha256_file(args.dev_manifest.with_suffix('.metadata.json')),
                   "labels_sha256": ev.sha256_file(LABELS)}
        if args.stage == "smoke":
            train = smallest_per_label(train, labels, 2)
            dev = smallest_per_label(dev, labels, 1)
            # Freeze the engineering subset so evaluator IDs/denominator match it.
            for split, rows, meta in (("train", train, train_meta), ("dev", dev, dev_meta)):
                manifest = output / f"smoke_{split}.jsonl"
                base.write_jsonl(manifest, rows)
                base.write_json(manifest.with_suffix('.metadata.json'), {
                    **meta, "manifest_sha256": ev.sha256_file(manifest),
                    "split_version": "w2-abc-engineering-smoke", "result_role": "engineering_only"})
            args = copy.copy(args)
            args.dev_manifest = output / "smoke_dev.jsonl"
        config = copy.deepcopy(config)
        config["seed"] = seed
        if args.stage == "smoke":
            config["optimization"]["epochs"] = 1
            config["experiment"]["scope"] = "smoke"
            config["result_role"] = "engineering_smoke_not_full_data_result"
        base.write_json(output / "config.json", config)
        record.update(data=binding, config=config, environment={"python": platform.python_version(),
                      "packages": {p: version(p) for p in ("torch", "torchaudio", "speechbrain", "numpy", "soundfile")},
                      "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
                      "gpu_count_used": int(torch.device(args.device).type == "cuda")})
        seed_everything(seed)
        record["environment"].update(deterministic_algorithms=True, matmul_tf32=False, cudnn_tf32=False,
                                     cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
        encoder = load_encoder(ae.FeatureConfig(layers=(25,), precision="fp32"), device=args.device, cache_dir=args.cache_root)
        model = AcousticDialectModel(encoder, args.architecture, len(labels), config["model"]["attention_dim"],
                                     config["model"]["hidden_dim"], config["model"]["dropout"]).to(encoder.device)
        record["encoder_provenance"] = encoder.provenance()
        record["encoder_state_before"] = reference.state_sha256(encoder)
        record["encoder_parameters_before"] = parameter_sha256(encoder)
        record["pooling_initial_sha256"] = reference.state_sha256(model.pooling)
        record["head_initial_sha256"] = reference.state_sha256(model.head)
        record["trainable_parameters"] = {"encoder": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                                           "pooling_and_head": sum(p.numel() for m in (model.pooling,model.head) for p in m.parameters())}
        resume = None
        if args.resume:
            resume = torch.load(args.resume, map_location="cpu", weights_only=True)
            if (resume["config"] != config or resume["data"] != binding or resume["labels"] != labels
                    or resume["source_sha256"] != record["source"]["sha256"]):
                raise ValueError("Resume checkpoint config/data/labels/source differs from this run")
            if resume["epoch"] >= config["optimization"]["epochs"]:
                raise ValueError("Resume checkpoint already exhausted the configured epoch budget")
            restore_model(model, resume["model"])
        initial = output / "initial_state.pt"
        atomic_save(initial, model_state(model))
        if args.stage != "smoke":
            preflight(model, optimizer_for(model, config), train, dev, args.audio_root, labels, output / "preflight.json")
            restore_model(model, torch.load(initial, map_location="cpu", weights_only=True))
            if args.stage == "preflight":
                record["status"] = "preflight_passed"
                initial.unlink()
                return 0
        train_ready, train_failures = audit_audio(train, args.audio_root, encoder.config, "train")
        dev_ready, dev_failures = audit_audio(dev, args.audio_root, encoder.config, "dev")
        failures = train_failures + dev_failures
        base.write_jsonl(output / "failures.jsonl", failures)
        if set(r["label"] for r in train_ready) != set(labels):
            raise ValueError("Usable training audio does not cover all 20 labels")
        training_ids_sha256 = base.sha256_text("\n".join(r["clip_id"] for r in train_ready))
        if resume and resume["training_ids_sha256"] != training_ids_sha256:
            raise ValueError("Usable training IDs differ from the resume checkpoint")
        # Gate changes are discarded; all full arms start from the declared pretrained state.
        if not resume:
            overfit_gate(model, train_ready, labels, args.audio_root, config, output / "overfit_check.json")
            restore_model(model, torch.load(initial, map_location="cpu", weights_only=True))
        else:
            base.write_json(output / "overfit_check.json", {"passed": True, "reused_from_verified_resume": str(args.resume)})
        initial.unlink()
        optimizer = optimizer_for(model, config)
        curve = [] if resume is None else resume["curve"]
        start_epoch = 0 if resume is None else resume["epoch"]
        if resume:
            optimizer.load_state_dict(resume["optimizer"])
            torch.set_rng_state(resume["torch_rng"])
            if encoder.device.type == "cuda":
                torch.cuda.set_rng_state(resume["cuda_rng"], encoder.device)
        else:
            seed_everything(seed)
        for epoch in range(start_epoch, config["optimization"]["epochs"]):
            tick = time.perf_counter()
            train_result = train_epoch(model, train_ready, labels, args.audio_root, optimizer, config, seed, epoch)
            predictions, epoch_failures = predict_rows(model, dev, args.audio_root, labels, f"epoch-{epoch+1}")
            metrics = ev.compute_metrics(ev.join_predictions(ev.validate_references(dev, labels, "dev"),
                                        ev.validate_prediction_fields(predictions, labels)), labels)
            curve.append({"epoch": epoch+1, "train": train_result,
                          "dev": {k: metrics[k] for k in ("accuracy", "macro_f1", "coverage", "failed")},
                          "seconds": time.perf_counter()-tick})
            base.write_jsonl(output / "learning_curve.jsonl", curve)
            saved = {"architecture": args.architecture, "labels": labels, "config": config, "data": binding,
                     "source_sha256": record["source"]["sha256"], "model": model_state(model),
                     "training_ids_sha256": training_ids_sha256,
                     "optimizer": optimizer.state_dict(), "epoch": epoch+1, "curve": curve,
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(encoder.device) if encoder.device.type == "cuda" else None}
            atomic_save(output / "checkpoint.pt", saved)
            base.progress(f"Epoch {epoch+1}: train loss {train_result['loss']:.4f}, dev macro-F1 {metrics['macro_f1']:.4f}")
        if not curve:
            raise ValueError("Resume checkpoint already exhausted the configured epoch budget")
        revision = f"acoustic-{args.architecture}:" + ev.sha256_file(output / "checkpoint.pt")[:16]
        predictions, final_failures = predict_rows(model, dev, args.audio_root, labels, revision)
        base.write_jsonl(output / "predictions.jsonl", predictions)
        base.write_jsonl(output / "failures.jsonl", train_failures + [{**f,"split":"dev"} for f in final_failures])
        metrics = evaluate_predictions(dev, predictions, labels, output / "evaluation", args)
        after = reference.state_sha256(encoder)
        parameter_after = parameter_sha256(encoder)
        record.update(encoder_state_after=after, encoder_updated=after != record["encoder_state_before"],
                      encoder_parameters_after=parameter_after,
                      encoder_parameter_updates_verified=parameter_after != record["encoder_parameters_before"],
                      epochs=len(curve), original_clip_exposures=len(train_ready)*len(curve),
                      optimization_updates=sum(c["train"]["updates"] for c in curve),
                      counts={"train_expected":len(train),"train_used":len(train_ready),"dev_expected":len(dev),"dev_successful":metrics["successful"]},
                      metrics={k:metrics[k] for k in ("accuracy","macro_f1","coverage")},
                      checkpoint_sha256=ev.sha256_file(output / "checkpoint.pt"))
        if args.architecture == "B" and record["encoder_updated"]:
            raise RuntimeError("B encoder state changed")
        if args.architecture == "C" and not record["encoder_parameter_updates_verified"]:
            raise RuntimeError("C encoder parameters did not change")
        record["status"] = "completed_with_failures" if train_failures or final_failures else "completed"
        return 2 if train_failures or final_failures else 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, message=str(error)[:1000])
        base.write_json(output / "failure.json", {"error_type":type(error).__name__,"message":str(error)[:1000]})
        raise
    finally:
        record["wall_seconds"] = time.perf_counter()-started
        record["peak_rss_mib"] = base.peak_rss_mib()
        record["run_directory_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if torch.device(args.device).type == "cuda":
            record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        base.write_json(output / "run_record.json", record)


def predict_checkpoint(args, load_encoder=ae.AcousticEncoder.from_pretrained):
    labels = ev.load_labels(LABELS)
    rows, split, meta = base.load_split(args.manifest, args.manifest.with_suffix('.metadata.json'), LABELS, labels)
    if split != "dev":
        raise ValueError("This entry point supports labeled dev diagnostics only; no test prediction")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in saved:  # Original A checkpoint contract.
        return acoustic_A_predict.main(["--checkpoint", str(args.checkpoint), "--manifest", str(args.manifest),
            "--metadata", str(args.manifest.with_suffix('.metadata.json')), "--audio-root", str(args.audio_root),
            "--output-root", str(args.output_root), "--cache-root", str(args.cache_root), "--device", args.device,
            "--output-dir", str(args.output_dir)], load_encoder=load_encoder)
    if saved["labels"] != labels or saved["architecture"] not in ("B","C"):
        raise ValueError("Checkpoint architecture/labels are invalid")
    if saved["source_sha256"] != source_identity()["sha256"]:
        raise ValueError("Use the training checkpoint's exact source version for inference")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    encoder = load_encoder(ae.FeatureConfig(layers=(25,), precision="fp32"), device=args.device, cache_dir=args.cache_root)
    m = saved["config"]["model"]
    model = AcousticDialectModel(encoder, saved["architecture"], len(labels), m["attention_dim"],m["hidden_dim"],m["dropout"]).to(encoder.device)
    restore_model(model, saved["model"])
    revision = f"acoustic-{saved['architecture']}:" + ev.sha256_file(args.checkpoint)[:16]
    predictions, failures = predict_rows(model, rows, args.audio_root, labels, revision)
    base.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    base.write_jsonl(args.output_dir / "failures.jsonl", failures)
    args.dev_manifest, args.architecture = args.manifest, saved["architecture"]
    evaluate_predictions(rows, predictions, labels, args.output_dir / "evaluation", args)
    base.write_json(args.output_dir / "prediction_record.json", {"fit_performed":False,"checkpoint_sha256":ev.sha256_file(args.checkpoint),
                    "manifest_sha256":meta["manifest_sha256"],"source":source_identity(),"failures":len(failures)})
    return 2 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("train","predict"):
        p = sub.add_parser(name)
        p.add_argument("--data-root",type=Path,required=True)
        p.add_argument("--audio-root",type=Path)
        p.add_argument("--cache-root",type=Path)
        p.add_argument("--output-root",type=Path)
        p.add_argument("--device",default="cuda")
        if name == "train":
            p.add_argument("--architecture",choices=("A","B","C"),required=True)
            p.add_argument("--stage",choices=("smoke","preflight","select","report"),default="smoke")
            p.add_argument("--runs-root",type=Path,required=True)
            p.add_argument("--train-manifest",type=Path,default=ROOT/"src/data/manifests/train.jsonl")
            p.add_argument("--dev-manifest",type=Path,default=ROOT/"src/data/manifests/dev.jsonl")
            p.add_argument("--resume",type=Path)
        else:
            p.add_argument("--checkpoint",type=Path,required=True)
            p.add_argument("--manifest",type=Path,required=True)
            p.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args(argv)
    args.audio_root = args.audio_root or args.data_root / "audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43"
    args.cache_root = args.cache_root or args.data_root / "cache/huggingface"
    args.output_root = args.output_root or args.data_root / "cache/week2_acoustic_A"
    if args.command == "predict":
        return predict_checkpoint(args)
    config = load_config(args.architecture)
    seeds = [17,29,43] if args.stage == "report" else [11]
    if args.resume:
        if args.architecture == "A" or args.stage not in ("select","report"):
            parser.error("Resume supports B/C select/report checkpoints at epoch boundaries")
        saved = torch.load(args.resume,map_location="cpu",weights_only=True)
        seeds = [saved["config"]["seed"]]
        if seeds[0] not in ([17,29,43] if args.stage=="report" else [11]):
            parser.error("Resume seed does not match the stage")
    for seed in seeds:
        seed_everything(seed)
        output = args.runs_root / args.architecture / f"{args.stage}-seed{seed}-{time.time_ns()}"
        output.parent.mkdir(parents=True,exist_ok=True)
        if args.architecture == "A":
            cfg = ROOT / "configs/week2_acoustic_A_smoke.json" if args.stage=="smoke" else CONFIGS/"A.json"
            train = ROOT/"data/manifests/train_pilot.jsonl" if args.stage=="smoke" else args.train_manifest
            dev = ROOT/"data/manifests/dev_pilot.jsonl" if args.stage=="smoke" else args.dev_manifest
            argv_a = ["--config",str(cfg),"--seed",str(seed),"--train-manifest",str(train),
                      "--train-metadata",str(train.with_suffix('.metadata.json')),"--dev-manifest",str(dev),
                      "--dev-metadata",str(dev.with_suffix('.metadata.json')),"--audio-root",str(args.audio_root),
                      "--output-root",str(args.output_root),"--cache-root",str(args.cache_root),"--device",args.device,
                      "--output-dir",str(output)]
            if args.stage=="preflight":argv_a += ["--preflight-only"]
            code = reference.main(argv_a)
        else:
            code = run_bc(args,config,seed,output)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
