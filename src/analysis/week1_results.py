"""Re-score the four frozen Week 1 releases and export paired error diagnostics.

Run from any directory: python src/analysis/week1_results.py --output-dir <new-dir>
Only the standard library and repository evaluator are needed. No model is fitted.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'evaluation'))
import evaluate as ev

RELEASES = {
    'mfcc': ('reports/mfcc_svm_pilot/predictions.jsonl',
             'f95153dacec70d805f09f32f759399f11b06590c8073a2394cd8f8171bbc3fac', 49, 0.23830908131117856),
    'acoustic': ('reports/acoustic_classifier_pilot/predictions.jsonl',
                 '3ef4b1fc9350535402fe54e81e7f07f08252bd5c7f1e1d3fe497366074fcb971', 67, 0.31489512472615855),
    'jais': ('reports/week1_evidence/jais_predictions.jsonl',
             'daaa9ec6513ac62bfd401517e16f29f5d575c752e48b2fd4040c8959c254f422', 143, 0.7045528323025294),
    'text': ('reports/week1_evidence/text_predictions.jsonl',
             '977e367b92b5235af17ede75ae170c8cb166ec205588f8ec8410f6b97cd6f08e', 50, 0.23642323859715164),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Frozen input changed: {path}; {actual} != {expected}')


def load_release(path, references, labels):
    rows = ev.read_jsonl(path)
    indexed = ev.index_by_clip_id(rows, str(path))
    if set(indexed) != set(references):
        raise ValueError(f'Pilot IDs differ: {path}')
    joined = ev.join_predictions(references, ev.validate_prediction_fields(rows, labels),
                                 allow_incomplete=False)
    metrics = ev.compute_metrics(joined, labels)
    if metrics['coverage'] != 1.0:
        raise ValueError(f'Released pilot must have complete successful coverage: {path}')
    return indexed, metrics


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-half, center+half]


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(output):
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest = ROOT / 'data/manifests/dev_pilot.jsonl'
    metadata = ROOT / 'data/manifests/dev_pilot.metadata.json'
    labels_path = ROOT / 'configs/labels.json'
    checked_hash(manifest, 'fb8684c35ef68e25834d5a4c82cddbd31044a899a63b665c41d3f9a179052d02')
    checked_hash(metadata, 'e85db634fb1487e3fcb241de8148b89c8b3d173fe54f60e183709c91f3fa65d7')
    checked_hash(labels_path, '5cf7d68237eab4a39790dda4faf59af6d31bff7c927564e55272e890206902c5')
    labels = ev.load_labels(labels_path)
    raw_refs = ev.read_jsonl(manifest)
    refs = ev.validate_references(raw_refs, labels, 'dev')
    raw_by_id = ev.index_by_clip_id(raw_refs, str(manifest))
    releases, results = {}, {}
    for name, (relative, digest, correct, f1) in RELEASES.items():
        path = ROOT / relative
        checked_hash(path, digest)
        releases[name], metrics = load_release(path, refs, labels)
        if metrics['correct'] != correct or not math.isclose(metrics['macro_f1'], f1, abs_tol=1e-14, rel_tol=0):
            raise ValueError(f'Released metrics disagree: {name}')
        results[name] = {
            k: metrics[k] for k in ['expected', 'correct', 'accuracy', 'macro_f1', 'coverage']
        }
        results[name].update(predictions=relative, sha256=digest,
                             accuracy_wilson95=wilson(correct, len(refs)))
    cases, correct_sets = [], {name: set() for name in releases}
    for clip_id in sorted(refs):
        row = raw_by_id[clip_id]
        case = {k: row[k] for k in ['clip_id', 'label', 'duration_sec', 'audio_path']}
        for name, predictions in releases.items():
            case[name] = predictions[clip_id]['predicted_label']
            if case[name] == row['label']:
                correct_sets[name].add(clip_id)
        cases.append(case)
    ids = set(refs)
    paired = {}
    for a, b in itertools.combinations(releases, 2):
        ca, cb = correct_sets[a], correct_sets[b]
        paired[f'{a}_vs_{b}'] = {
            'both_correct': len(ca & cb), 'first_only_correct': len(ca-cb),
            'second_only_correct': len(cb-ca), 'both_wrong': len(ids-(ca|cb)),
            'same_wrong_label': sum(c[a] == c[b] != c['label'] for c in cases),
            'oracle_correct_diagnostic_only': len(ca|cb),
        }
    confusions, examples, per_dialect = {}, [], []
    for name in releases:
        counts = Counter((c['label'], c[name]) for c in cases if c['label'] != c[name])
        top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:5]
        confusions[name] = [{'true': a, 'predicted': b, 'count': count} for (a, b), count in top]
        for (a, b), count in top:
            candidates = [c for c in cases if c['label'] == a and c[name] == b]
            for c in candidates[:2]:
                examples.append(dict(model=name, confusion=f'{a}->{b}', count=count, **c))
        for label in labels:
            truth = [c for c in cases if c['label'] == label]
            tp = sum(c[name] == label for c in truth)
            predicted = sum(c[name] == label for c in cases)
            precision, recall = tp/predicted if predicted else 0, tp/len(truth)
            per_dialect.append(dict(model=name, label=label, support=len(truth), correct=tp,
                                    predicted=predicted, precision=precision, recall=recall,
                                    f1=2*precision*recall/(precision+recall) if precision+recall else 0))
    audit = json.loads((ROOT / 'reports/audit.json').read_text())
    acoustic = json.loads((ROOT / 'reports/acoustic_classifier_pilot/extraction_run_record.json').read_text())
    jais = json.loads((ROOT / 'reports/jais_pilot_metrics.json').read_text())
    total = audit['summary']['clips']
    audio_sec = audit['summary']['duration_hours'] * 3600
    throughput = acoustic['features']['extracted_audio_sec'] / acoustic['resources']['feature_extraction_sec']
    # Train includes 20 cache hits and load/I/O, so this is a planning bracket, not pure generation time.
    seconds_per_new = [jais['splits'][s]['wall_seconds'] /
                       (jais['splits'][s]['expected'] - jais['splits'][s]['cache_hits']) for s in ['train', 'dev']]
    acoustic_hours = audio_sec / throughput / 3600
    compute = {
        'full_clips': total, 'full_audio_hours': audio_sec/3600,
        'acoustic_measured_audio_sec_per_wall_sec': throughput,
        'acoustic_full_extraction_hours_nominal': acoustic_hours,
        'acoustic_planning_hours_1x_to_2x_nominal': [acoustic_hours, 2*acoustic_hours],
        'jais_seconds_per_new_clip_bracket': seconds_per_new,
        'jais_full_cold_wall_hours_bracket': [total*s/3600 for s in seconds_per_new],
        'jais_remaining_clips_reusing_700': total-700,
        'jais_remaining_wall_hours_bracket': [(total-700)*s/3600 for s in seconds_per_new],
        'gpu_price_per_hour': None,
        'pricing_note': 'No agreed rate. Cost = single-GPU occupied wall hours * rate; rates must match the measured GPU. No A100/H100 scaling measured.',
        'raw_bytes': audit['download_bytes'], 'wav_bytes': audit['summary']['extracted_bytes'],
        'feature_float32_bytes_all_26_layers': total * 26 * 848 * 4,
        'feature_float32_bytes_final_layer_only': total * 848 * 4,
    }
    summary = {
        'scope': 'in_source_fixed_pilot_not_official_test',
        'split_version': 'w1-07-pilot-v1-e61cd569ac7b',
        'manifest_sha256': sha(manifest), 'labels_sha256': sha(labels_path),
        'metrics': results, 'paired_correctness': paired, 'top_directed_confusions': confusions,
        'all_four_wrong': len(ids-set.union(*correct_sets.values())),
        'acoustic_correct_when_jais_and_text_wrong': len(correct_sets['acoustic']-(correct_sets['jais']|correct_sets['text'])),
        'all_four_oracle_correct_diagnostic_only': len(set.union(*correct_sets.values())),
        'compute': compute,
        'uncertainty_note': 'Wilson intervals are descriptive clip-level intervals; clustered recordings and pilot selection limit population inference.',
    }
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    fields = ['clip_id', 'label', 'duration_sec', 'audio_path', *releases]
    write_csv(output / 'per_clip.csv', cases, fields)
    write_csv(output / 'error_examples.csv', examples, ['model', 'confusion', 'count', *fields])
    write_csv(output / 'per_dialect.csv', per_dialect,
              ['model', 'label', 'support', 'correct', 'predicted', 'precision', 'recall', 'f1'])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output_dir), indent=2))
