"""Offline Task 13 export checks; no model, audio or GPU results are claimed."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

from src.experiments import jais_pilot as pilot
from src.models import jais_adapter as jais

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'configs/labels.json'


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'dev.jsonl'
        self.metadata = self.root / 'dev.metadata.json'
        rows = [{'clip_id': str(i), 'label': 'MSA', 'split': 'dev',
                 'source_id': 'DO_NOT_COPY', 'audio_path': f'{i}.wav'} for i in range(7)]
        pilot.write_jsonl(self.manifest, rows)
        jais.atomic_json(self.metadata, {
            'schema_version': '1.0', 'dataset_version': 'fixture', 'split_version': 'fixture',
            'manifest_sha256': jais.sha256_file(self.manifest),
            'labels_sha256': jais.sha256_file(LABELS), 'evaluation_label_sets': {'dev': ['MSA']}})
        records = []
        for i in range(7):
            item = SimpleNamespace(text='مرحبا', language='msa', refused=False, truncated=False)
            if i == 1:
                item.language = 'unknown'
            elif i == 2:
                item.truncated = True
            elif i == 3:
                item.truncated = None
            elif i == 4:
                item.refused = True
            elif i == 5:
                records.append(jais.failure(str(i), jais.MODEL_REVISION, 'audio_error', 'read_failed'))
                continue
            elif i == 6:
                item.text = '[LB] مرحبا'
            records.append(jais.parsed_record(str(i), jais.MODEL_REVISION, '[msa] raw', item))
        self.run = self.root / 'run'
        record = {
            'status': 'completed_with_failures', 'model_id': jais.MODEL_ID,
            'model_revision': jais.MODEL_REVISION, 'expected': 7,
            'manifest_sha256': jais.sha256_file(self.manifest),
            'metadata_sha256': jais.sha256_file(self.metadata),
            'labels_sha256': jais.sha256_file(LABELS),
            'dataset_version': 'fixture', 'split_version': 'fixture',
            'counts': {'expected': 7, 'successful': 4, 'failed': 3,
                       'cache_hits': 0, 'model_batches': 4, 'new_outputs': 6},
            'wall_seconds': 7.0}
        jais.write_outputs(self.run, records, record)

    def tearDown(self):
        self.temp.cleanup()

    def export(self, output='export'):
        return pilot.export_run(self.manifest, self.metadata, LABELS, 'dev',
                                self.run, self.root / output)

    def read(self, path):
        return [json.loads(x) for x in path.read_text().splitlines()]

    def rehash(self):
        record = jais.read_json(self.run / 'run_record.json')
        for name in ['predictions', 'raw_outputs']:
            record[name + '_sha256'] = jais.sha256_file(self.run / (name + '.jsonl'))
        jais.atomic_json(self.run / 'run_record.json', record)

    def test_export_preserves_ids_and_separates_labels_from_text(self):
        result = self.export()
        rows = self.read(self.root / 'export/transcripts.jsonl')
        self.assertEqual([r['clip_id'] for r in rows], [str(i) for i in range(7)])
        self.assertEqual(result['expected'], 7)
        self.assertEqual(result['accuracy'], 4 / 7)
        self.assertEqual([r['transcript_status'] for r in rows],
                         ['ok', 'ok', 'truncated', 'truncation_unknown', 'refused', 'audio_error', 'tag_contamination'])
        self.assertEqual([r['usable_for_text_model'] for r in rows], [True, True, False, False, False, False, False])
        self.assertEqual(rows[0]['transcript'], 'مرحبا')
        for row in rows:
            self.assertFalse({'label', 'predicted_label', 'native_dialect', 'source_id', 'audio_path'} & row.keys())
        self.assertEqual(len(self.read(self.root / 'export/targets.jsonl')), 7)
        summary = jais.read_json(self.root / 'export/summary.json')
        self.assertIsNone(summary['cold_end_to_end_clips_per_second'])  # failed run
        self.assertIsNone(summary['model_generation_gpu_seconds'])

    def test_unknown_tag_does_not_remove_usable_transcript(self):
        self.export()
        row = self.read(self.root / 'export/transcripts.jsonl')[1]
        self.assertEqual(row['transcript_status'], 'ok')
        self.assertTrue(row['usable_for_text_model'])

    def test_tampered_outputs_rejected_before_export(self):
        with (self.run / 'raw_outputs.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.export()
        self.assertFalse((self.root / 'export').exists())

    def test_duplicate_and_missing_raw_ids_rejected(self):
        original = self.read(self.run / 'raw_outputs.jsonl')
        for rows in [original[:-1], original + original[:1]]:
            pilot.write_jsonl(self.run / 'raw_outputs.jsonl', rows)
            self.rehash()
            with self.assertRaises(ValueError):
                self.export()

    def test_prediction_raw_disagreement_rejected(self):
        rows = self.read(self.run / 'predictions.jsonl')
        rows[0]['predicted_label'] = 'LEB'
        pilot.write_jsonl(self.run / 'predictions.jsonl', rows)
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.export()

    def test_wrong_release_or_unfinished_run_rejected(self):
        original = jais.read_json(self.run / 'run_record.json')
        for patch in [{'status': 'running'}, {'manifest_sha256': 'wrong'}, {'expected': 6}]:
            jais.atomic_json(self.run / 'run_record.json', {**original, **patch})
            with self.assertRaises(ValueError):
                self.export()

    def test_cli_and_shared_evaluator_agree(self):
        subprocess.run([sys.executable, str(ROOT / 'src/experiments/jais_pilot.py'),
                        'export', '--manifest', str(self.manifest), '--metadata', str(self.metadata),
                        '--split', 'dev', '--run-dir', str(self.run),
                        '--output-dir', str(self.root / 'cli-export')], check=True, capture_output=True)
        subprocess.run([sys.executable, str(ROOT / 'src/evaluation/evaluate.py'),
                        '--manifest', str(self.manifest), '--metadata', str(self.metadata),
                        '--predictions', str(self.run / 'predictions.jsonl'), '--split', 'dev',
                        '--model-name', 'offline-fixture', '--output-dir', str(self.root / 'evaluation')],
                       check=True, capture_output=True, cwd=ROOT)
        exported = jais.read_json(self.root / 'cli-export/direct_dialect_metrics.json')
        evaluated = jais.read_json(self.root / 'evaluation/metrics.json')
        for key in ['accuracy', 'macro_f1', 'coverage', 'expected', 'successful', 'failed', 'missing']:
            self.assertEqual(exported[key], evaluated[key])

    def test_existing_exports_not_overwritten(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_smoke_uses_one_real_frozen_train_clip_per_label(self):
        result = pilot.prepare_smoke(ROOT / 'data/manifests/train_pilot.jsonl',
                                    ROOT / 'data/manifests/train_pilot.metadata.json',
                                    LABELS, self.root / 'smoke')
        self.assertEqual(result['clips'], 20)
        rows = self.read(self.root / 'smoke/smoke.jsonl')
        self.assertEqual(len({r['label'] for r in rows}), 20)
        self.assertTrue(all(r['split'] == 'train' for r in rows))
        pilot.checked_manifest(self.root / 'smoke/smoke.jsonl',
                               self.root / 'smoke/smoke.metadata.json', LABELS, 'train')

    def test_empty_parse_failure_and_spoken_dialect_name_policy(self):
        item = SimpleNamespace(text='  ', language='msa', refused=False, truncated=False)
        row = jais.parsed_record('x', jais.MODEL_REVISION, 'raw', item)
        self.assertEqual(pilot.transcript_row(row)['transcript_status'], 'empty')
        row = jais.failure('x', jais.MODEL_REVISION, 'inference_error', 'parse_failed')
        self.assertEqual(pilot.transcript_row(row)['transcript_status'], 'inference_error')
        item.text = 'أنا لبناني'
        row = jais.parsed_record('x', jais.MODEL_REVISION, 'raw', item)
        self.assertEqual(pilot.transcript_row(row)['transcript_status'], 'ok')


if __name__ == '__main__':
    unittest.main()
