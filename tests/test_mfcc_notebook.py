"""Check the actual notebook setup cell without Colab or network access."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / 'notebooks/W1_11_MFCC_SVM_Baseline.ipynb').read_text())
        self.source = ''.join(notebook['cells'][2]['source'])
        assignments = {n.targets[0].id: n.value.value for n in ast.parse(self.source).body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and isinstance(n.value, ast.Constant)}
        self.pin = assignments['PINNED_COMMIT']
        self.assertRegex(self.pin, r'^[a-f0-9]{40}$')
        self.assertNotIn('SOURCE_REF', assignments)

    def execute(self, root, actual):
        scope = {'Path': Path, 'DATA_ROOT': root}
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(tempfile, 'mkdtemp', return_value=str(root / 'source')), \
             patch.object(subprocess, 'run', side_effect=run), \
             patch.object(subprocess, 'check_output', return_value=actual + '\n'):
            exec(compile(self.source, '<mfcc-notebook-setup>', 'exec'), scope)
        return calls

    def test_fetches_immutable_reviewed_source_without_branch_name(self):
        with tempfile.TemporaryDirectory() as td:
            calls = self.execute(Path(td), self.pin)
        fetches = [a for a in calls if 'fetch' in a]
        self.assertEqual(len(fetches), 1)
        self.assertEqual(fetches[0][-2:], ['origin', self.pin])
        self.assertTrue(any(a[-3:] == ['--quiet', '--detach', 'FETCH_HEAD'] for a in calls))

    def test_rejects_checkout_of_another_commit(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, 'Source mismatch'):
                self.execute(Path(td), '0' * 40)


if __name__ == '__main__':
    unittest.main()
