"""Verify a Task 13 evidence ZIP against this checkout's frozen pilots.

No model/audio loading. Reproduces exports and checks the full-dev denominator,
input/output/source hashes, metric records and confusion CSVs. Does not verify
speech transcription accuracy or independently measure the recorded GPU run.

Usage: python tests/check_jais_pilot_evidence.py /path/to/task13_evidence.zip
"""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.jais_pilot import export_run
from src.models.jais_adapter import sha256_file


def verify_bundle(bundle):
    results = {}
    with tempfile.TemporaryDirectory() as td, ZipFile(bundle) as archive:
        work = Path(td)
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive entry names')
        for entry in archive.infolist():
            destination = (work / entry.filename).resolve()
            if not destination.is_relative_to(work.resolve()):
                raise ValueError('Unsafe archive path')
            if entry.file_size > 20_000_000:
                raise ValueError('Unexpectedly large evidence entry')
        if sum(i.file_size for i in archive.infolist()) > 50_000_000:
            raise ValueError('Unexpectedly large evidence archive')
        archive.extractall(work)
        for split, expected in [('train', 500), ('dev', 200)]:
            manifest = ROOT / f'data/manifests/{split}_pilot.jsonl'
            metadata = manifest.with_suffix('.metadata.json')
            run_dir = work / f'{split}_run'
            original = work / f'{split}_export'
            regenerated = work / f'{split}_regenerated'
            result = export_run(manifest, metadata, ROOT / 'configs/labels.json',
                                split, run_dir, regenerated)
            if result['expected'] != expected:
                raise ValueError(f'{split}: unexpected pilot size')
            for path in regenerated.iterdir():
                if path.read_bytes() != (original / path.name).read_bytes():
                    raise ValueError(f'{split}: export differs: {path.name}')
            run = json.loads((run_dir / 'run_record.json').read_text())
            if run['adapter_code_sha256'] != sha256_file(ROOT / 'src/models/jais_adapter.py'):
                raise ValueError('Adapter source does not match this checkout')
            metrics = json.loads((original / 'evaluation/metrics.json').read_text())
            direct = json.loads((regenerated / 'direct_dialect_metrics.json').read_text())
            # All computed fields, not only the rounded headline metrics.
            from src.evaluation.evaluate import compute_metrics, join_predictions, load_labels, read_jsonl, validate_prediction_fields, validate_references
            labels = load_labels(ROOT / 'configs/labels.json')
            recomputed = compute_metrics(join_predictions(
                validate_references(read_jsonl(manifest), labels, split),
                validate_prediction_fields(read_jsonl(run_dir / 'predictions.jsonl'), labels),
                allow_incomplete=False), labels)
            for key, value in recomputed.items():
                if metrics.get(key) != value or direct.get(key) != value:
                    raise ValueError(f'{split}: inconsistent metric: {key}')
            if metrics['evaluator_revision'] != sha256_file(ROOT / 'src/evaluation/evaluate.py'):
                raise ValueError('Evaluator source differs')
            for key, path in [('manifest_sha256', manifest), ('metadata_sha256', metadata),
                              ('labels_sha256', ROOT / 'configs/labels.json'),
                              ('predictions_sha256', run_dir / 'predictions.jsonl')]:
                if metrics.get(key) != sha256_file(path):
                    raise ValueError(f'{split}: evaluation input hash differs: {key}')
            matrix = metrics['confusion_matrix']
            csv_rows = list(csv.reader(io.StringIO((original / 'evaluation/confusion_matrix.csv').read_text())))
            expected_csv = [['true_label', *matrix['column_labels']]] + [
                [label, *map(str, counts)] for label, counts in zip(matrix['row_labels'], matrix['counts'], strict=True)]
            if csv_rows != expected_csv:
                raise ValueError(f'{split}: confusion CSV differs')
            results[split] = {k: metrics[k] for k in ['expected', 'correct', 'accuracy', 'macro_f1', 'coverage', 'failed', 'missing']}
    return {'bundle_sha256': sha256_file(bundle), 'verified': results}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(verify_bundle(Path(sys.argv[1])), indent=2))
