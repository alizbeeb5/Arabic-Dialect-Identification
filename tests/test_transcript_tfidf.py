"""Meaningful fit/predict, leakage, failure-policy and serialization checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
from src.models import transcript_tfidf as text
from src.models.jais_adapter import atomic_json, sha256_file
from src.experiments.jais_pilot import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / 'configs/labels.json').read_text())['label_order']
CONFIG = json.loads((ROOT / 'configs/transcript_tfidf.json').read_text())


def record(clip_id, transcript, status='ok'):
    return {'clip_id': clip_id, 'transcript': transcript, 'transcript_status': status,
            'usable_for_text_model': status == 'ok', 'refused': status == 'refused',
            'truncated': status == 'truncated'}


class TextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [record(f'{i}-{j}', f'unique_{label} language expression {label} {j}')
                       for i, label in enumerate(LABELS) for j in range(3)]
        cls.targets = {r['clip_id']: LABELS[int(r['clip_id'].split('-')[0])] for r in cls.records}
        cls.bundle, cls.diagnostics = text.fit_model(cls.records, cls.targets, LABELS, CONFIG, {'fixture': True})

    def test_development_vocabulary_never_fitted(self):
        vocab = {n: dict(v.vocabulary_) for n, v in self.bundle['vectorizer'].transformer_list}
        text.predict_records(self.bundle, [record('new', 'developmentonlyneverfitted')], 'fixture')
        for name, vectorizer in self.bundle['vectorizer'].transformer_list:
            self.assertEqual(vectorizer.vocabulary_, vocab[name])
        self.assertNotIn('developmentonlyneverfitted', vocab['word'])

    def test_probabilities_are_reordered_from_native_classes(self):
        predictions, _ = text.predict_records(self.bundle, self.records[:4], 'fixture')
        native = self.bundle['classifier'].predict_proba(
            self.bundle['vectorizer'].transform([r['transcript'] for r in self.records[:4]]))
        classes = list(self.bundle['classifier'].classes_)
        self.assertNotEqual(classes, LABELS)
        for row, scores in zip(predictions, native, strict=True):
            np.testing.assert_allclose(row['scores'], [scores[classes.index(c)] for c in LABELS])
            self.assertEqual(row['score_labels'], LABELS)
            self.assertAlmostEqual(sum(row['scores']), 1)

    def test_missing_and_refused_dev_rows_use_training_prior_without_dropping(self):
        records = [record('empty', None, 'empty'), record('refusal', None, 'refused'),
                   record('partial', 'available words', 'truncated')]
        predictions, diagnostics = text.predict_records(self.bundle, records, 'fixture')
        self.assertEqual([r['clip_id'] for r in predictions], ['empty', 'refusal', 'partial'])
        self.assertEqual([r['prediction_source'] for r in diagnostics],
                         ['training_class_prior', 'training_class_prior', 'text'])
        for row in predictions[:2]:
            np.testing.assert_allclose(row['scores'], self.bundle['class_prior'])
            self.assertEqual(row['predicted_label'], 'MSA')  # canonical tie, not alphabetic

    def test_partial_text_rule_is_identical_during_fit_and_predict(self):
        records = self.records + [record('partial-train', 'partial useful speech', 'truncated'),
                                  record('refused-train', None, 'refused')]
        targets = {**self.targets, 'partial-train': 'MSA', 'refused-train': 'MSA'}
        bundle, diag = text.fit_model(records, targets, LABELS, CONFIG, {})
        self.assertEqual(diag['retained_training_clips'], len(self.records) + 1)
        self.assertEqual(diag['excluded_training_ids'], ['refused-train'])
        self.assertEqual(diag['retained_training_class_counts']['MSA'], 4)
        _, diagnostics = text.predict_records(bundle, records[-2:], 'fixture')
        self.assertEqual(diagnostics[0]['prediction_source'], 'text')

    def test_marker_contamination_and_missing_class_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Generated tags'):
            text.predict_records(self.bundle, [record('contaminated', '[LB] hello')], 'fixture')
        with self.assertRaisesRegex(ValueError, 'Every canonical class'):
            text.fit_model(self.records[3:], self.targets, LABELS, CONFIG, {})

    def test_reloaded_model_produces_identical_scores(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.joblib'
            joblib.dump(self.bundle, path)
            predictions, _ = text.predict_records(self.bundle, self.records, 'fixture')
            reloaded, _ = text.predict_records(joblib.load(path), self.records, 'fixture')
            self.assertEqual(predictions, reloaded)

    def test_export_hashes_and_gold_input_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, metadata = root / 'train.jsonl', root / 'train.metadata.json'
            refs = [{'clip_id': r['clip_id'], 'label': self.targets[r['clip_id']], 'split': 'train'}
                    for r in self.records]
            write_jsonl(manifest, refs)
            atomic_json(metadata, {'schema_version': '1.0', 'dataset_version': 'fixture',
                'split_version': 'fixture', 'manifest_sha256': sha256_file(manifest),
                'labels_sha256': sha256_file(ROOT / 'configs/labels.json'),
                'evaluation_label_sets': {'train': LABELS}})
            write_jsonl(root / 'transcripts.jsonl', self.records)
            write_jsonl(root / 'targets.jsonl', [{'clip_id': r['clip_id'], 'label': r['label']} for r in refs])
            summary = {'manifest_sha256': sha256_file(manifest), 'metadata_sha256': sha256_file(metadata),
                'labels_sha256': sha256_file(ROOT / 'configs/labels.json'), 'split': 'train',
                'split_version': 'fixture', 'dataset_version': 'fixture', 'expected': len(refs),
                'transcripts.jsonl_sha256': sha256_file(root / 'transcripts.jsonl'),
                'targets.jsonl_sha256': sha256_file(root / 'targets.jsonl')}
            atomic_json(root / 'summary.json', summary)
            args = (root, manifest, metadata, ROOT / 'configs/labels.json', 'train')
            records, targets, labels, identity = text.load_export(*args)
            self.assertEqual(records, self.records)
            self.assertEqual(targets, self.targets)
            contaminated = [dict(r) for r in self.records]
            contaminated[0]['label'] = 'MSA'
            write_jsonl(root / 'transcripts.jsonl', contaminated)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                text.load_export(*args)
            summary['transcripts.jsonl_sha256'] = sha256_file(root / 'transcripts.jsonl')
            atomic_json(root / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'no label/source fields'):
                text.load_export(*args)

    def test_cli_rejects_training_on_development(self):
        result = subprocess.run([sys.executable, str(ROOT / 'src/models/transcript_tfidf.py'),
            'train', '--export-dir', 'unused', '--manifest', 'unused', '--metadata', 'unused',
            '--output-dir', 'unused', '--split', 'dev'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Training accepts only', result.stderr)


if __name__ == '__main__':
    unittest.main()
