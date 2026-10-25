"""Check the W1-12 notebook's source and run cells without Colab, GPU or network access."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/W1_12_Acoustic_Classifier.ipynb'
COLAB_REPO = "Path('/content/Arabic-Dialect-Identification')"


def code_cells():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    return [''.join(c['source']) for c in cells if c['cell_type'] == 'code']


def cell_with(text):
    return next(source for source in code_cells() if text in source)


class NotebookTests(unittest.TestCase):
    def test_metadata_requests_gpu_and_cells_are_plain_python(self):
        metadata = json.loads(NOTEBOOK.read_text())['metadata']
        self.assertEqual(metadata['accelerator'], 'GPU')
        for source in code_cells():
            ast.parse(source)
            self.assertFalse(any(line.lstrip().startswith(('!', '%')) for line in source.splitlines()))

    def run_cell(self, mode):
        source = cell_with('subprocess.Popen')
        commands = []

        class Process:
            def __init__(self, argv, **kwargs):
                commands.append(argv)
                self.stdout = iter(['done\n'])

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as td:
            scope = {'sys': sys, 'subprocess': subprocess, 'MODE': mode, 'REPO_ROOT': ROOT,
                     'AUDIO_ROOT': Path(td) / 'audio', 'OUTPUT_ROOT': Path(td) / 'output',
                     'CACHE_ROOT': Path(td) / 'hf', 'RUN_ROOT': Path(td) / 'run'}
            with patch.object(subprocess, 'Popen', Process), patch('builtins.print'):
                exec(compile(source, '<run-cell>', 'exec'), scope)
        self.assertEqual(scope['exit_code'], 0)
        return commands[0]

    def test_mode_selects_overfit_gate_or_full_run(self):
        gate = self.run_cell('overfit_check')
        self.assertIn('--overfit-check-only', gate)
        self.assertEqual(gate[gate.index('--device') + 1], 'cuda')
        self.assertNotIn('--overfit-check-only', self.run_cell('full'))

    def test_zip_source_unpacks_and_records_base_commit_and_dirty_state(self):
        source = cell_with("SOURCE = 'zip'")
        self.assertIn(COLAB_REPO, source)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            checkout = td / 'checkout'
            (checkout / 'src/experiments').mkdir(parents=True)
            (checkout / 'src/experiments/acoustic_classifier.py').write_text('# committed\n')
            git = lambda *a: subprocess.run(['git', '-C', str(checkout), *a], check=True, capture_output=True,
                                            text=True).stdout.strip()
            git('init', '--quiet')
            git('add', '.')
            git('-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '--quiet', '-m', 'base')
            base_commit = git('rev-parse', 'HEAD')
            (checkout / 'src/experiments/acoustic_classifier.py').write_text('# uncommitted edit\n')
            archive = td / 'Arabic-Dialect-Identification-w1-12.zip'
            with zipfile.ZipFile(archive, 'w') as zf:
                for path in checkout.rglob('*'):
                    if path.is_file():
                        zf.write(path, path.relative_to(checkout))
            colab = types.ModuleType('google.colab')
            colab.files = types.SimpleNamespace(upload=lambda: {archive.name: archive.read_bytes()})
            google = types.ModuleType('google')
            google.colab = colab
            scope = {'Path': Path, 'DATA_ROOT': td / 'data', 'MODE': 'overfit_check'}
            cwd = os.getcwd()
            try:
                os.chdir(td)
                with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
                     patch('builtins.print'):
                    exec(compile(source.replace(COLAB_REPO, f"Path({str(td / 'repo')!r})"), '<source-cell>',
                                 'exec'), scope)
            finally:
                os.chdir(cwd)
        self.assertEqual(scope['SOURCE_COMMIT'], base_commit)
        self.assertIn('acoustic_classifier.py', scope['SOURCE_STATUS'])
        self.assertRegex(scope['SOURCE_SHA256'], r'^[a-f0-9]{64}$')
        self.assertTrue(scope['RUN_ROOT'].name.endswith('_overfit_check'))

    def test_github_source_requires_a_pinned_commit(self):
        source = cell_with("SOURCE = 'zip'").replace("SOURCE = 'zip'", "SOURCE = 'github'", 1)
        with tempfile.TemporaryDirectory() as td:
            scope = {'Path': Path, 'DATA_ROOT': Path(td), 'MODE': 'full'}
            source = source.replace(COLAB_REPO, f"Path({str(Path(td) / 'repo')!r})")
            with self.assertRaisesRegex(RuntimeError, 'PINNED_COMMIT'):
                exec(compile(source, '<source-cell>', 'exec'), scope)


if __name__ == '__main__':
    unittest.main()
