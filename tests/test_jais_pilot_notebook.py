"""Exercise the notebook's actual subprocess/checkout functions without a GPU."""
import ast
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NotebookCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        notebook = json.loads((ROOT / 'notebooks/W1_13_Jais_Pilot.ipynb').read_text())
        source = ''.join(notebook['cells'][2]['source'])
        tree = ast.parse(source)
        definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
        self.scope = dict(Path=Path, subprocess=subprocess, sys=sys, datetime=datetime,
                          timezone=timezone, LOG_ROOT=self.root / 'logs', REPO_ROOT=self.root)
        self.scope['LOG_ROOT'].mkdir()
        exec(compile(definitions, '<notebook-functions>', 'exec'), self.scope)
        self.source_tree = tree

    def test_streams_stdout_stderr_and_persists_log(self):
        seen_before_child_exit = self.root / 'seen'
        class Display(io.StringIO):
            def write(inner, value):
                if 'first-progress' in value:
                    seen_before_child_exit.touch()
                return super().write(value)
        display = Display()
        # Child waits for the parent's display to observe progress before exiting.
        # A capture-only implementation would time out and return failure.
        script = ("import sys,time; from pathlib import Path; "
                  "print('first-progress',flush=True); "
                  "print('stderr-progress',file=sys.stderr,flush=True); "
                  f"p=Path({str(seen_before_child_exit)!r}); "
                  "deadline=time.monotonic()+5\n"
                  "while not p.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
                  "sys.exit(0 if p.exists() else 9)")
        with contextlib.redirect_stdout(display):
            code = self.scope['run_logged']([sys.executable, '-u', '-c', script], log_name='live')
        self.assertEqual(code, 0)
        self.assertIn('first-progress', display.getvalue())
        self.assertIn('stderr-progress', display.getvalue())
        log = next(self.scope['LOG_ROOT'].glob('*-live.log')).read_text()
        self.assertIn('first-progress', log)
        self.assertIn('stderr-progress', log)

    def test_exit_two_is_preserved_and_other_failures_point_to_log(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.scope['run_logged']([sys.executable, '-c', 'raise SystemExit(2)'],
                                             allowed=(0, 2), log_name='allowed')
            self.assertEqual(code, 2)
            with self.assertRaisesRegex(RuntimeError, r'exited 7; see .*failure.log'):
                self.scope['run_logged']([sys.executable, '-c', "print('details'); raise SystemExit(7)"],
                                         log_name='failure')
        self.assertIn('details', next(self.scope['LOG_ROOT'].glob('*-failure.log')).read_text())

    def test_checkout_works_after_original_branch_is_deleted(self):
        repo = self.root / 'source'
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Notebook test'], check=True)
        subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.invalid'], check=True)
        (repo / 'value').write_text('verified version')
        subprocess.run(['git', '-C', str(repo), 'add', 'value'], check=True)
        subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'verified source'], check=True)
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['git', '-C', str(repo), 'branch', 'temporary-task-branch'], check=True)
        subprocess.run(['git', '-C', str(repo), 'branch', '-D', 'temporary-task-branch'], check=True, capture_output=True)
        destination = self.root / 'checkout'
        with contextlib.redirect_stdout(io.StringIO()):
            actual = self.scope['checkout_source'](str(repo), destination, commit)
        self.assertEqual(actual, commit)
        self.assertEqual((destination / 'value').read_text(), 'verified version')
        self.assertEqual(subprocess.run(['git', '-C', str(destination), 'symbolic-ref', '-q', 'HEAD'],
                                        capture_output=True).returncode, 1)

    def test_notebook_pins_verified_commit_and_child_python_is_unbuffered(self):
        assignments = {n.targets[0].id: n.value.value for n in self.source_tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and isinstance(n.value, ast.Constant)}
        self.assertEqual(assignments['PINNED_SOURCE_COMMIT'], '26b36593c7b4857b34a0b5f864c409f6d546be91')
        captured = []
        self.scope['run_logged'] = lambda argv, **kwargs: captured.append((argv, kwargs)) or 0
        self.scope['command']('src/models/jais_adapter.py', ['--batch-size', '2'], allowed=(0, 2))
        self.assertEqual(captured[0][0][:2], [sys.executable, '-u'])
        self.assertEqual(captured[0][1]['allowed'], (0, 2))


if __name__ == '__main__':
    unittest.main()
