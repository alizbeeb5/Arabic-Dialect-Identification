"""Train/reload the Task 14 text baseline from verified Task 13 exports."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits

from src.evaluation import evaluate as ev
from src.experiments.jais_pilot import checked_manifest, TAG_PATTERN, write_jsonl
from src.models.jais_adapter import atomic_json, read_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
TEXT_FIELDS = {'clip_id', 'transcript', 'transcript_status', 'usable_for_text_model', 'refused', 'truncated'}
STATUSES = {'ok', 'truncated', 'truncation_unknown', 'empty', 'refused',
            'tag_contamination', 'audio_error', 'inference_error'}


def validate_config(config):
    expected_policy = {'accepted_statuses': ['ok', 'truncated'],
                       'excluded_training_text': 'drop_and_report',
                       'unavailable_dev_text': 'retained_training_class_prior',
                       'use_partial_text': True,
                       'fit_vectorizer_on': 'accepted_training_transcripts_only'}
    if config.get('input_policy') != expected_policy:
        raise ValueError('Unsupported text policy; preregister policy changes explicitly')
    if config.get('selection') != {'hyperparameter_search': False, 'configuration_fixed_before_dev_scoring': True}:
        raise ValueError('This baseline has a fixed configuration, no dev tuning')
    if set(config.get('features', {})) != {'word', 'character'}:
        raise ValueError('Specify exactly word and character feature branches')
    return config


def load_export(export_dir, manifest, metadata, labels_path, split):
    rows, references, labels, meta = checked_manifest(manifest, metadata, labels_path, split)
    export = Path(export_dir)
    summary = read_json(export / 'summary.json')
    for key, expected in {'manifest_sha256': sha256_file(manifest),
                          'metadata_sha256': sha256_file(metadata),
                          'labels_sha256': sha256_file(labels_path), 'split': split,
                          'split_version': meta['split_version'],
                          'dataset_version': meta['dataset_version'], 'expected': len(rows)}.items():
        if summary.get(key) != expected:
            raise ValueError(f'Transcript release mismatch: {key}')
    for name in ['transcripts.jsonl', 'targets.jsonl']:
        if summary.get(name + '_sha256') != sha256_file(export / name):
            raise ValueError(f'Export hash mismatch: {name}')
    records = ev.index_by_clip_id(ev.read_jsonl(export / 'transcripts.jsonl'), 'transcripts')
    targets = ev.index_by_clip_id(ev.read_jsonl(export / 'targets.jsonl'), 'targets')
    if set(records) != set(references) or set(targets) != set(references):
        raise ValueError('Export IDs must exactly match the frozen manifest')
    for clip_id, record in records.items():
        if set(record) != TEXT_FIELDS:
            raise ValueError('Text records must contain only the transcript schema; no label/source fields')
        if record['transcript_status'] not in STATUSES:
            raise ValueError('Unknown transcript status')
        if record['transcript'] is not None and not isinstance(record['transcript'], str):
            raise ValueError('Transcript must be text or null')
        if type(record['usable_for_text_model']) is not bool:
            raise ValueError('Invalid usability flag')
        for key in ['refused', 'truncated']:
            if record[key] is not None and type(record[key]) is not bool:
                raise ValueError(f'Invalid {key} flag')
        if targets[clip_id] != {'clip_id': clip_id, 'label': references[clip_id]['label']}:
            raise ValueError('Targets disagree with the frozen manifest')
    identity = {key: summary[key] for key in ['manifest_sha256', 'metadata_sha256', 'labels_sha256',
                'split_version', 'dataset_version', 'transcripts.jsonl_sha256', 'targets.jsonl_sha256']}
    identity['summary_sha256'] = sha256_file(export / 'summary.json')
    return [records[r['clip_id']] for r in rows], {k: r['label'] for k, r in references.items()}, labels, identity


def available_text(record):
    """Explicit override: use known partial text, never treat it as missing."""
    text = record['transcript']
    status = record['transcript_status']
    if status not in {'ok', 'truncated'} or record['refused'] is not False:
        return None
    if record['truncated'] is not (status == 'truncated'):
        raise ValueError('Text status and truncation flag disagree')
    if not isinstance(text, str) or not text.strip():
        return None
    if TAG_PATTERN.search(text) or '<|' in text:
        raise ValueError('Generated tags must not enter text features')
    return text


def fit_model(records, targets, labels, config, training_identity):
    validate_config(config)
    accepted = [(r['clip_id'], available_text(r)) for r in records]
    retained = [(clip_id, text) for clip_id, text in accepted if text is not None]
    if not retained:
        raise ValueError('No available training text')
    counts = Counter(targets[clip_id] for clip_id, _ in retained)
    if set(counts) != set(labels):
        raise ValueError('Every canonical class needs available training text')
    features = FeatureUnion([(name, TfidfVectorizer(**{
        **options, 'ngram_range': tuple(options['ngram_range'])}))
        for name, options in config['features'].items()])
    classifier = LogisticRegression(**config['classifier'], random_state=config['seed'])
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        matrix = features.fit_transform([text for _, text in retained])
        classifier.fit(matrix, [targets[clip_id] for clip_id, _ in retained])
    prior = np.array([counts[label] / len(retained) for label in labels])
    bundle = {'schema_version': '1.0', 'vectorizer': features, 'classifier': classifier,
              'label_order': labels, 'class_prior': prior, 'config': config,
              'training_input': training_identity}
    diagnostics = {'expected_training_clips': len(records), 'retained_training_clips': len(retained),
                   'excluded_training_ids': [clip_id for clip_id, text in accepted if text is None],
                   'retained_training_class_counts': {c: counts[c] for c in labels},
                   'training_transcript_status_counts': dict(Counter(r['transcript_status'] for r in records)),
                   'feature_counts': {name: len(v.vocabulary_) for name, v in features.transformer_list},
                   'training_matrix_shape': list(matrix.shape),
                   'optimizer_iterations': classifier.n_iter_.tolist(),
                   'fit_seconds': time.perf_counter() - start}
    return bundle, diagnostics


def predict_records(bundle, records, model_revision):
    labels = bundle['label_order']
    classifier = bundle['classifier']
    # sklearn's alphabetical native order is not the benchmark order.
    native = list(classifier.classes_)
    if set(native) != set(labels):
        raise ValueError('Checkpoint classes do not match the canonical vocabulary')
    order = [native.index(c) for c in labels]
    texts = [available_text(r) for r in records]
    available = [i for i, text in enumerate(texts) if text is not None]
    scores = np.tile(bundle['class_prior'], (len(records), 1))
    if available:
        with threadpool_limits(limits=1):
            matrix = bundle['vectorizer'].transform([texts[i] for i in available])
            scores[available] = classifier.predict_proba(matrix)[:, order]
    if not np.isfinite(scores).all() or np.any(scores < 0) or not np.allclose(scores.sum(1), 1):
        raise ValueError('Invalid categorical probability vectors')
    predictions, diagnostics = [], []
    for record, text, row in zip(records, texts, scores, strict=True):
        predictions.append({'clip_id': record['clip_id'], 'predicted_label': labels[int(np.argmax(row))],
                            'model_revision': model_revision, 'status': 'ok', 'scores': row.tolist(),
                            'score_labels': labels, 'score_type': 'probabilities'})
        diagnostics.append({'clip_id': record['clip_id'], 'transcript_status': record['transcript_status'],
                            'prediction_source': 'text' if text is not None else 'training_class_prior'})
    ev.validate_prediction_fields(predictions, labels)
    return predictions, diagnostics


def code_identity():
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
        return {'git_commit': commit, 'git_dirty': dirty}
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': None, 'git_dirty': None}


def run(args):
    labels_path = ROOT / 'configs/labels.json'
    records, targets, labels, identity = load_export(args.export_dir, args.manifest, args.metadata, labels_path, args.split)
    output = args.output_dir
    if output.exists():
        raise ValueError('Use a new output directory')
    if args.command == 'train':
        config = validate_config(read_json(args.config))
        bundle, diagnostics = fit_model(records, targets, labels, config, identity)
        output.mkdir(parents=True, exist_ok=False)
        model_path = output / 'model.joblib'
        joblib.dump(bundle, model_path, compress=3)
        atomic_json(output / 'training_diagnostics.json', diagnostics)
        atomic_json(output / 'config.json', config)
    else:
        model_path = args.model
        bundle = joblib.load(model_path)
        if bundle['label_order'] != labels:
            raise ValueError('Model label order differs from this release')
        output.mkdir(parents=True, exist_ok=False)
    revision = 'sha256:' + sha256_file(model_path)
    predictions, diagnostics = predict_records(bundle, records, revision)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'prediction_diagnostics.jsonl', diagnostics)
    joined = ev.join_predictions(ev.validate_references(ev.read_jsonl(args.manifest), labels, args.split),
                                 ev.validate_prediction_fields(predictions, labels), allow_incomplete=False)
    metrics = ev.compute_metrics(joined, labels)
    metrics.update(model_name='transcript-tfidf-logreg', model_revision=revision, split=args.split,
                   manifest_sha256=sha256_file(args.manifest), metadata_sha256=sha256_file(args.metadata),
                   labels_sha256=sha256_file(labels_path), predictions_sha256=sha256_file(output / 'predictions.jsonl'),
                   evaluator_revision=sha256_file(ev.__file__))
    atomic_json(output / 'metrics.json', metrics)
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                   for p in ['numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']}}
    record = {'command': args.command, 'model_revision': revision, 'input': identity,
              'training_input': bundle['training_input'], 'config': bundle['config'],
              'environment': environment, 'source_sha256': sha256_file(__file__), **code_identity(),
              'expected': len(records), 'prediction_sources': dict(Counter(d['prediction_source'] for d in diagnostics)),
              'probability_semantics': 'multinomial logistic regression; not independently calibrated',
              'prediction_command': [sys.executable, *sys.argv]}
    atomic_json(output / 'run_record.json', record)
    print(json.dumps({'output_dir': str(output), 'accuracy': metrics['accuracy'],
                      'macro_f1': metrics['macro_f1'], 'coverage': metrics['coverage'],
                      'prediction_sources': record['prediction_sources']}, indent=2))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['train', 'predict']:
        sub = commands.add_parser(name)
        sub.add_argument('--export-dir', type=Path, required=True)
        sub.add_argument('--manifest', type=Path, required=True)
        sub.add_argument('--metadata', type=Path, required=True)
        sub.add_argument('--output-dir', type=Path, required=True)
        sub.add_argument('--split', choices=['train', 'dev'], required=True)
        if name == 'train':
            sub.add_argument('--config', type=Path, default=ROOT / 'configs/transcript_tfidf.json')
        else:
            sub.add_argument('--model', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'train' and args.split != 'train':
        parser.error('Training accepts only the training split')
    try:
        run(args)
    except (ValueError, OSError, KeyError, TypeError, ConvergenceWarning) as error:
        parser.exit(1, f'Text baseline error: {error}\n')


if __name__ == '__main__':
    main()
