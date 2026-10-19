# Shared data contracts
Version: 0.1 | Date: 2026-10-19 | Status: draft for team review
Applies to [EEC-5 / W1-01](https://linear.app/eece490/issue/EEC-5/w1-01-define-the-benchmark-protocol-and-shared-data-contracts).
Companion files: [protocol.md](protocol.md) and [../configs/labels.json](../configs/labels.json).

## 1. Encoding and identity
Use UTF-8 JSONL: one JSON object per nonempty line, with no enclosing array. Examples below are synthetic; they do not assert the existence of actual clips or trained checkpoints.

IDs are nonempty, stable strings unique across the dataset release. Namespacing IDs by dataset is recommended. Join predictions to references by clip_id, never by incidental file enumeration order. Manifest order is retained separately for official export.

Each immutable manifest release has a companion metadata JSON object recording schema_version, dataset_version, split_version, manifest_sha256, labels_sha256, audio_root, evaluation_label_sets, selection_split, and source provenance. Record actual checksum values when releases are built; do not use illustrative placeholders in experiment records. evaluation_label_sets defines the supported labels for each accessible labeled evaluation split. A hidden test's set is not inferred for model selection.

## 2. Audio manifest
All fields below are required keys. Null is allowed only where explicitly stated.

| Field | Type | Rule |
|---|---|---|
| clip_id | string | Nonempty and unique across the release |
| audio_path | string | Relative path under declared audio_root; no parent-directory traversal |
| label | string or null | Code from labels.json; null only for unlabeled/hidden-label data |
| split | string | train, dev, or test |
| source_id | string or null | Namespaced original recording/episode identifier; null means unknown and is an audit limitation |
| speaker_id | string or null | Namespaced known speaker identifier; null if unknown |
| domain | string | Controlled release vocabulary, e.g. podcast, broadcast, telephone, read_speech, unknown |
| duration_sec | number | Finite and strictly positive, measured from stored audio |
| sample_rate_hz | integer | Strictly positive, measured from stored audio |

Example:
```json
{"clip_id":"clip_000001","audio_path":"audio/clip_000001.wav","label":"LEB","split":"dev","source_id":"podcast:recording_0042","speaker_id":null,"domain":"podcast","duration_sec":8.4,"sample_rate_hz":16000}
```

The sample rate is descriptive, not a universal 16-kHz requirement. Record model-specific resampling, channel handling, cropping/windowing, and aggregation in the run configuration. Do not drop long or unreadable clips silently.

Train and scored development data require known labels. A separate unlabeled-data use must be declared explicitly. Hidden test labels should be inaccessible to model code. Source IDs should be recovered during ingestion where possible; unknown provenance cannot establish source independence. Dataset ingestion must audit these cases before a source-independent split is claimed.

Optional extensions include source_dataset, original_clip_id, source_uri, audio_sha256, channel_count, and recording offsets. Define any extension's type in the release metadata. Do not add annotation-sensitive personal information unnecessarily.

### Leakage boundary
Models consume permitted audio and declared transformations derived from it. Labels are targets; filenames, paths, source metadata, speaker IDs, and split fields must not enter predictive features or prompts. Transcript branches use transcripts derived from permitted audio. Dataset grouping metadata is for auditing and splitting only.

## 3. Prediction records
| Field | Type | Rule |
|---|---|---|
| clip_id | string | Matches exactly one expected manifest record |
| predicted_label | string or null | Valid code for status=ok; null for failure |
| model_revision | string | Immutable checkpoint/content identifier; for ensembles, identifies a manifest of component revisions and weights |
| status | string | ok, audio_error, or inference_error |
| scores | array[number] | Optional; exactly 20 finite entries if supplied |
| score_labels | array[string] | Required with scores; exact label_order from labels.json |
| score_type | string | Required with scores: logits, probabilities, or log_probabilities |
| error_code | string | Optional short machine-readable reason on failures |

Successful hard-label prediction:
```json
{"clip_id":"clip_000001","predicted_label":"LEB","model_revision":"example-only-checkpoint-v1","status":"ok"}
```

Failure:
```json
{"clip_id":"clip_000002","predicted_label":null,"model_revision":"example-only-checkpoint-v1","status":"audio_error","error_code":"decode_failed"}
```

Successful prediction with logits:
```json
{"clip_id":"clip_000001","predicted_label":"LEB","model_revision":"example-only-checkpoint-v1","status":"ok","scores":[0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0],"score_labels":["MSA","BAH","TUN","ALG","EGY","IRA","JOR","KSA","KUW","LEB","LIB","MAU","MOR","OMA","PAL","QAT","SUD","SYR","UAE","YEM"],"score_type":"logits"}
```

Replace illustrative model_revision values with real immutable identifiers when running experiments. A run sidecar records run_id, code commit, model revisions, configs, seeds, manifest/split hashes, label-config hash, and inference command. All rows in one prediction file refer to the same declared system revision.

### Score semantics
- When scores are absent, omit scores, score_labels, and score_type together.
- Failure rows omit all score fields; never fabricate a vector for failed inference.
- For probabilities, every value is in [0,1] and the sum differs from 1 by at most 1e-5.
- For log_probabilities, every value is at most 1e-6 and abs(logsumexp(scores)) is at most 1e-5.
- Logits need only be finite real numbers.
- Reject NaN and infinities for all score types. Do not silently clamp invalid values.
- Reorder native model outputs before writing. score_labels must equal the canonical order exactly.
- With scores, predicted_label equals the label at argmax(scores); ties choose the lowest canonical index. Store the final decision scores if calibration/combination changes the prediction.
- A hard-label-only system remains valid for accuracy/F1. It is not eligible for C_avg without valid full class scores.

## 4. Validation and evaluator behavior
Separate format validation from performance evaluation.

| Condition | Action |
|---|---|
| Duplicate clip_id | Reject prediction artifact; do not choose one row |
| Unexpected clip_id | Reject artifact |
| Invalid label/status/field type | Reject artifact |
| status=ok with null label | Reject artifact |
| Failure with non-null label or score fields | Reject artifact |
| Missing expected clip_id | Flag incomplete contract; count as an incorrect prediction in diagnostic headline accuracy, with missing count visible |
| Explicit failure | Count as incorrect; preserve its reason |
| Invalid/nonfinite scores | Reject artifact; fix and rerun validation |
| No scores provided | Accuracy/F1 supported; C_avg unavailable |
| No reference label | No local accuracy/F1; export only if permitted |

For a complete-run report, require every expected clip to appear, including explicit failures. A partial diagnostic result must be labeled incomplete, and cannot silently substitute for a valid final submission.

Accuracy uses the complete expected labeled denominator. Failure/missing rows produce false negatives for their true labels. They do not create false positives for an invented real class. Confusion matrices retain the 20 dialect prediction columns plus a separate failure/missing column. Apply the macro-F1/support convention in protocol.md.

C_avg must not silently exclude failures or missing scores. Report unavailable unless the verified benchmark scorer explicitly defines a compatible full-set failure policy.

Evaluation summaries record dataset/split versions, prediction hash, model revision, evaluator revision, expected/successful/failed/missing counts, coverage, accuracy, supported-class macro-F1 with its exact label set, per-class results, confusion matrix, and C_avg value or its unavailability reason.

## 5. Label mapping
The machine-readable mapping lives in ../configs/labels.json.

| ID | Code | Meaning |
|---:|---|---|
| 0 | MSA | Modern Standard Arabic |
| 1 | BAH | Bahrain |
| 2 | TUN | Tunisia |
| 3 | ALG | Algeria |
| 4 | EGY | Egypt |
| 5 | IRA | Iraq |
| 6 | JOR | Jordan |
| 7 | KSA | Saudi Arabia |
| 8 | KUW | Kuwait |
| 9 | LEB | Lebanon |
| 10 | LIB | Libya |
| 11 | MAU | Mauritania |
| 12 | MOR | Morocco |
| 13 | OMA | Oman |
| 14 | PAL | Palestine |
| 15 | QAT | Qatar |
| 16 | SUD | Sudan |
| 17 | SYR | Syria |
| 18 | UAE | United Arab Emirates |
| 19 | YEM | Yemen |

## 6. Official submission boundary
Our JSONL contracts are internal. The [organizer notebook](https://colab.research.google.com/drive/1oxF7CyEGJwhDMWmAoeg-OYSDh6kLsUfW), Submission Format, demonstrates:
- logits.tsv: one row per sample, 20 tab-separated class scores in canonical order, no header.
- predictions.tsv: one zero-based predicted class ID per sample, no header.
- submission.zip: contains both files.

Export must join by clip_id and then follow the official evaluation sample order. Internal clip IDs are not extra TSV columns. Confirm the actual platform contract and score semantics before export; the notebook requests logits, and any conversion from other score types must be explicit and validated. An official sample-order artifact is still required.

For export, reject missing predictions, explicit inference failures, missing score vectors, and row-count/order mismatches rather than inventing submission values. A scoreless internal baseline can still participate in local accuracy/F1 comparisons.

## 7. Acceptance checklist
- [x] Example manifest and prediction rows specified.
- [x] Complete label mapping specified.
- [x] Metadata leakage boundary specified.
- [x] Failure and score semantics specified.
- [ ] Implement validator/evaluator in W1-08 / EEC-12.
- [ ] Populate and audit real manifests in ingestion/split tasks.
- [ ] Confirm actual platform scoring, sample order, and current submission requirements.
- [ ] Review this contract with all three members and record a different teammate's review.

This document defines the contract; it does not claim that a validator, dataset audit, or official submission has already been implemented.

