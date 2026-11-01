"""Execute the notebook's report checks without downloading audio or importing torch."""
import ast
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class CasablancaNotebookTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.notebook = json.loads((ROOT / "notebooks/W2_Casablanca_Validation_Prep.ipynb").read_text())
        source = self.source(1)
        definitions = ast.Module(body=[n for n in ast.parse(source).body
                                      if isinstance(n, ast.FunctionDef)], type_ignores=[])
        self.scope = {"json": json, "CODE_ROOT": ROOT, "LOG_DIR": self.path,
                      "DATA_ROOT": self.path, "sys": sys}
        exec(compile(definitions, "notebook-functions", "exec"), self.scope)
        self.manifest = ROOT / "data/manifests/casablanca_validation.jsonl"
        self.valid = {"status": "passed", "checked": 50, "failures": [],
                      "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                      "split_version": json.loads(self.manifest.with_suffix(".metadata.json").read_text())["split_version"]}

    def source(self, index):
        value = self.notebook["cells"][index]["source"]
        return value if isinstance(value, str) else "".join(value)

    def execute_preprocessing(self, exit_code=0, report=None):
        def run_logged(argv, name):
            if report is not None:
                output = Path(argv[argv.index("--output") + 1])
                output.write_text(json.dumps(report))
            return exit_code
        self.scope["run_logged"] = run_logged
        with patch.dict(sys.modules, {"torchaudio": types.ModuleType("torchaudio")}), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(self.source(7), self.scope)

    def test_nonzero_exit_stops_even_when_old_report_passes(self):
        with self.assertRaisesRegex(AssertionError, "Preprocessing failed"):
            self.execute_preprocessing(exit_code=1, report=self.valid)

    def test_absent_report_rejected_despite_committed_success_report(self):
        self.assertTrue((ROOT / "reports/casablanca_validation_audit/preprocessing_check.json").is_file())
        with self.assertRaisesRegex(ValueError, "no preprocessing report"):
            self.execute_preprocessing()

    def test_failed_empty_and_stale_reports_rejected(self):
        for update in ({"status": "failed"}, {"checked": 0}, {"failures": [{}]},
                       {"manifest_sha256": "stale"}, {"split_version": "stale"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.execute_preprocessing(report={**self.valid, **update})

    def test_valid_attempt_packaged_instead_of_historical_report(self):
        self.execute_preprocessing(report=self.valid)
        current = self.scope["PREPROCESSING_REPORT"]
        self.assertEqual(self.scope["preprocessing"], self.valid)
        writes = []
        class Archive:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def write(self, path, name): writes.append((path, name))
        self.scope.update(zipfile=types.SimpleNamespace(ZipFile=Archive, ZIP_DEFLATED=8),
                          files=types.SimpleNamespace(download=lambda _: None), Path=Path)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.source(9), self.scope)
        expected = "reports/casablanca_validation_audit/preprocessing_check.json"
        self.assertEqual([(p, name) for p, name in writes if name == expected], [(current, expected)])
        self.assertNotIn(ROOT / expected, [p for p, _ in writes])

    def test_packaging_rechecks_report_after_a_failed_attempt(self):
        self.execute_preprocessing(report=self.valid)
        self.scope["PREPROCESSING_REPORT"].write_text(json.dumps({**self.valid, "status": "failed"}))
        with self.assertRaises(ValueError):
            exec(self.source(9), self.scope)


if __name__ == "__main__":
    unittest.main()
