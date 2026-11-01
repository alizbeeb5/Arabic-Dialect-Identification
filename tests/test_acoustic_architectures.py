"""B/C gradients, masking, artifact/reload and epoch-resume checks; tiny random encoder only."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from src.models import acoustic_encoder as ae
from src.models.acoustic_architectures import AcousticDialectModel, MaskedAttentionPooling
from src.experiments import acoustic_abc as abc
from test_mfcc_svm_baseline import write_split

ROOT = Path(__file__).resolve().parents[1]
TINY = ae.Architecture(d_model=32, nhead=4, num_layers=2, d_ffn=64)


def tiny_encoder(config, device=None, cache_dir=None):
    torch.manual_seed(0)
    config = ae.FeatureConfig(layers=(3,))
    return ae.AcousticEncoder(config, TINY, device="cpu")


class ArchitectureTests(unittest.TestCase):
    def test_attention_excludes_nan_padding_and_rejects_empty_masks(self):
        torch.manual_seed(2)
        pooling = MaskedAttentionPooling(4, 8)
        x = torch.randn(2, 7, 4)
        mask = ae.frame_mask(torch.tensor([7, 3]), 7)
        expected = pooling(x[1:2, :3], torch.ones(1, 3, dtype=torch.bool))
        x[1, 3:] = torch.nan
        self.assertTrue(torch.allclose(pooling(x, mask)[1], expected[0], atol=1e-6))
        with self.assertRaises(ValueError):
            pooling(x, torch.zeros_like(mask))

    def test_bc_share_heads_and_only_c_updates_encoder(self):
        waves = [torch.randn(3200) * 0.1, torch.randn(4800) * 0.1]
        states = {}
        for arm in ("B", "C"):
            encoder = tiny_encoder(None)
            model = AcousticDialectModel(encoder, arm)
            states[arm] = {"pooling":copy.deepcopy(model.pooling.state_dict()),"head":copy.deepcopy(model.head.state_dict())}
            before = abc.reference.state_sha256(encoder)
            optimizer = abc.optimizer_for(model, abc.load_config(arm))
            model.train()
            torch.nn.functional.cross_entropy(model(waves), torch.tensor([1, 2])).backward()
            summary = abc.gradient_summary(model)
            self.assertGreater(summary["pooling_parameters_with_nonzero_gradient"], 0)
            self.assertGreater(summary["head_parameters_with_nonzero_gradient"], 0)
            self.assertEqual(summary["encoder_parameters_with_nonzero_gradient"] > 0, arm == "C")
            optimizer.step()
            self.assertEqual(abc.reference.state_sha256(encoder) != before, arm == "C")
            if arm == "B":
                self.assertFalse(encoder.training)
                self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        for part in ("pooling","head"):
            for name, value in states["B"][part].items():
                self.assertTrue(torch.equal(value, states["C"][part][name]))

    def test_trainable_final_matches_frozen_encode_in_eval(self):
        encoder = tiny_encoder(None)
        waves = [torch.randn(4000)*0.1,torch.randn(2000)*0.1]
        with torch.no_grad():
            frames, mask, lengths = encoder.encode_final(waves)
            stacked, other_mask, other_lengths = encoder.encode(waves)
        torch.testing.assert_close(frames, stacked[0], atol=0, rtol=0)
        self.assertTrue(torch.equal(mask, other_mask))
        self.assertTrue(torch.equal(lengths, other_lengths))

    def test_configs_hold_bc_capacity_and_exposure_fixed(self):
        b, c = abc.load_config("B"), abc.load_config("C")
        self.assertEqual(b["model"],c["model"])
        self.assertEqual(b["optimization"],c["optimization"])
        self.assertEqual(b["optimization"]["epochs"],abc.load_config("A")["optimization"]["epochs"])
        self.assertEqual(b["model"]["hidden_dim"],256)
        self.assertEqual(b["model"]["dropout"],0.2)
        self.assertEqual(b["experiment"]["data"],c["experiment"]["data"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, _, self.train_rows = write_split(self.root,"train",per_label=2)
        self.dev, _, self.dev_rows = write_split(self.root,"dev",per_label=1)
        self.args = argparse.Namespace(architecture="B",stage="smoke",train_manifest=self.train,
            dev_manifest=self.dev,audio_root=self.root/"audio",device="cpu",cache_root=self.root/"hf",resume=None)

    def config(self,arm):
        config = abc.load_config(arm)
        config["experiment"]["scope"] = "smoke"  # synthetic fixtures cannot satisfy full-release guard
        config["overfit_check"].update(epochs=1,min_accuracy=0.0,max_loss_ratio=2.0)
        config["optimization"].update(epochs=2,effective_batch_size=16)
        return config

    def run_arm(self,arm,name,stage="smoke",resume=None):
        args=copy.copy(self.args)
        args.architecture,args.stage,args.resume=arm,stage,resume
        out=self.root/name
        with contextlib.redirect_stdout(io.StringIO()):
            code=abc.run_bc(args,self.config(arm),11,out,load_encoder=tiny_encoder)
        return out,code

    def test_b_and_c_end_to_end_with_independent_prediction_reload(self):
        for arm in ("B","C"):
            out,code=self.run_arm(arm,arm)
            self.assertEqual(code,0)
            saved=torch.load(out/"checkpoint.pt",weights_only=True)
            self.assertEqual("encoder" in saved["model"],arm=="C")
            record=json.loads((out/"run_record.json").read_text())
            self.assertEqual(record["encoder_updated"],arm=="C")
            self.assertEqual(record["encoder_parameter_updates_verified"],arm=="C")
            self.assertEqual(record["original_clip_exposures"],40)
            self.assertEqual(record["optimization_updates"],3)  # final partial accumulation is retained
            rows=abc.ev.read_jsonl(out/"predictions.jsonl")
            self.assertEqual(len(rows),20)
            pred=argparse.Namespace(checkpoint=out/"checkpoint.pt",manifest=out/"smoke_dev.jsonl",
                  audio_root=self.root/"audio",cache_root=self.root/"hf",device="cpu",output_dir=self.root/f"infer-{arm}")
            with patch.object(abc,"train_epoch",side_effect=AssertionError("prediction trained")):
                self.assertEqual(abc.predict_checkpoint(pred,load_encoder=tiny_encoder),0)
            other=abc.ev.read_jsonl(pred.output_dir/"predictions.jsonl")
            np.testing.assert_allclose([p["scores"] for p in rows],[p["scores"] for p in other],atol=1e-6,rtol=0)
            self.assertTrue((out/"evaluation/confusion_matrix.csv").is_file())

    def test_interruption_resumes_exact_c_epoch_and_optimizer_state(self):
        uninterrupted,_=self.run_arm("C","uninterrupted",stage="select")
        actual_save=abc.atomic_save
        def interrupt_after_first(path,value):
            if Path(path).name=="checkpoint.pt" and value["epoch"]==2:
                raise KeyboardInterrupt("simulated stopped job")
            return actual_save(path,value)
        with patch.object(abc,"atomic_save",side_effect=interrupt_after_first), self.assertRaises(KeyboardInterrupt):
            self.run_arm("C","stopped",stage="select")
        checkpoint=self.root/"stopped/checkpoint.pt"
        self.assertEqual(torch.load(checkpoint,weights_only=True)["epoch"],1)
        resumed,code=self.run_arm("C","resumed",stage="select",resume=checkpoint)
        self.assertEqual(code,0)
        a=torch.load(uninterrupted/"checkpoint.pt",weights_only=True)
        b=torch.load(resumed/"checkpoint.pt",weights_only=True)
        for component in ("encoder","pooling","head"):
            for name,tensor in a["model"][component].items():
                torch.testing.assert_close(tensor,b["model"][component][name],atol=0,rtol=0)

    def test_unreadable_dev_remains_in_full_denominator(self):
        (self.root/"audio"/self.dev_rows[0]["audio_path"]).write_bytes(b"wrong audio")
        out,code=self.run_arm("B","failure")
        self.assertEqual(code,2)
        metrics=json.loads((out/"evaluation/metrics.json").read_text())
        self.assertEqual((metrics["expected"],metrics["failed"]),(20,1))
        self.assertEqual(abc.ev.read_jsonl(out/"predictions.jsonl")[0]["status"],"audio_error")

    def test_full_contract_rejects_pilot_before_encoder_load(self):
        args=copy.copy(self.args)
        args.stage="select"
        out=self.root/"invalid-full"
        with self.assertRaises(ValueError), patch.object(abc.ae.AcousticEncoder,"from_pretrained",side_effect=AssertionError("loaded")):
            abc.run_bc(args,abc.load_config("B"),11,out)
        self.assertEqual(json.loads((out/"run_record.json").read_text())["status"],"failed")

    def test_gate_failure_is_durable_and_prevents_training(self):
        config=self.config("B")
        config["overfit_check"].update(min_accuracy=1.0,max_loss_ratio=0.0)
        out=self.root/"failed-gate"
        with self.assertRaisesRegex(RuntimeError,"overfit check failed"):
            abc.run_bc(self.args,config,11,out,load_encoder=tiny_encoder)
        self.assertFalse((out/"checkpoint.pt").exists())
        self.assertFalse(json.loads((out/"overfit_check.json").read_text())["passed"])
        self.assertTrue((out/"failure.json").exists())

    def test_common_predict_rejects_test_before_a_or_bc_model_load(self):
        rows=copy.deepcopy(self.dev_rows)
        for row in rows:
            row["split"]="test"
        test=self.root/"test.jsonl"
        abc.base.write_jsonl(test,rows)
        metadata=json.loads(self.dev.with_suffix('.metadata.json').read_text())
        metadata.update(split="test",manifest_sha256=abc.ev.sha256_file(test),
                        evaluation_label_sets={"test":abc.ev.load_labels(abc.LABELS)})
        abc.base.write_json(test.with_suffix('.metadata.json'),metadata)
        args=argparse.Namespace(manifest=test,checkpoint=self.root/"not-even-opened.pt")
        with self.assertRaisesRegex(ValueError,"no test prediction"):
            abc.predict_checkpoint(args)

    def test_longest_training_preflight_really_updates_and_preserves_failure(self):
        model=AcousticDialectModel(tiny_encoder(None),"C")
        optimizer=abc.optimizer_for(model,self.config("C"))
        before=abc.parameter_sha256(model.encoder)
        output=self.root/"preflight.json"
        abc.preflight(model,optimizer,self.train_rows,self.dev_rows,self.root/"audio",
                      abc.ev.load_labels(abc.LABELS),output)
        result=json.loads(output.read_text())
        self.assertTrue(result["passed"])
        self.assertNotEqual(before,abc.parameter_sha256(model.encoder))
        expected=max(self.train_rows,key=lambda r:r["duration_sec"])["clip_id"]
        self.assertEqual(result["checks"][0]["clip_id"],expected)
        with patch.object(abc,"load_audio",side_effect=RuntimeError("simulated longest-clip OOM")), \
                self.assertRaisesRegex(RuntimeError,"preflight failed"):
            abc.preflight(model,optimizer,self.train_rows,self.dev_rows,self.root/"audio",
                          abc.ev.load_labels(abc.LABELS),output)
        result=json.loads(output.read_text())
        self.assertFalse(result["passed"])
        self.assertIn("OOM",result["checks"][0]["message"])

    def test_development_labels_do_not_change_c_weights(self):
        original,_=self.run_arm("C","labels-original",stage="select")
        rows=copy.deepcopy(self.dev_rows)
        labels=abc.ev.load_labels(abc.LABELS)
        for r in rows:
            r["label"]=labels[(labels.index(r["label"])+1)%len(labels)]
        abc.base.write_jsonl(self.dev,rows)
        meta_path=self.dev.with_suffix('.metadata.json')
        meta=json.loads(meta_path.read_text())
        meta["manifest_sha256"]=abc.ev.sha256_file(self.dev)
        abc.base.write_json(meta_path,meta)
        permuted,_=self.run_arm("C","labels-permuted",stage="select")
        a=torch.load(original/"checkpoint.pt",weights_only=True)
        b=torch.load(permuted/"checkpoint.pt",weights_only=True)
        for component in ("encoder","pooling","head"):
            for name,tensor in a["model"][component].items():
                torch.testing.assert_close(tensor,b["model"][component][name],atol=0,rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
