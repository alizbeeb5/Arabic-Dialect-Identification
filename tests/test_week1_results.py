"""Checks that protect the paired comparison from misaligned or changed evidence."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('week1_results', ROOT / 'src/analysis/week1_results.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class EvidenceAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.labels = analysis.ev.load_labels(ROOT / 'configs/labels.json')
        self.refs = analysis.ev.validate_references(
            analysis.ev.read_jsonl(ROOT / 'data/manifests/dev_pilot.jsonl'), self.labels, 'dev')
        self.rows = analysis.ev.read_jsonl(ROOT / 'reports/week1_evidence/jais_predictions.jsonl')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'predictions.jsonl'

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_row_order_does_not_change_score(self):
        self.write(list(reversed(self.rows)))
        _, metrics = analysis.load_release(self.path, self.refs, self.labels)
        self.assertEqual(metrics['correct'], 143)
        self.assertAlmostEqual(metrics['macro_f1'], 0.7045528323025294)

    def test_missing_extra_and_duplicate_ids_reject(self):
        extra = dict(self.rows[0], clip_id='not-in-frozen-pilot')
        for rows in [self.rows[:-1], [*self.rows, extra], [*self.rows, self.rows[0]]]:
            with self.subTest(length=len(rows)):
                self.write(rows)
                with self.assertRaises(ValueError):
                    analysis.load_release(self.path, self.refs, self.labels)

    def test_modified_frozen_file_rejects(self):
        self.write(self.rows)
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            analysis.checked_hash(self.path, '0'*64)

    def test_all_releases_and_paired_counts(self):
        result = analysis.build(Path(self.temp.name) / 'analysis')
        for key, expected in [('acoustic_vs_jais', (53, 14, 90, 43)),
                              ('jais_vs_text', (34, 109, 16, 41))]:
            actual = result['paired_correctness'][key]
            self.assertEqual(tuple(actual[k] for k in ['both_correct', 'first_only_correct',
                                                       'second_only_correct', 'both_wrong']), expected)
        self.assertEqual(result['all_four_wrong'], 33)
        self.assertEqual(result['acoustic_correct_when_jais_and_text_wrong'], 6)
        self.assertIsNone(result['compute']['gpu_price_per_hour'])


if __name__ == '__main__':
    unittest.main()
