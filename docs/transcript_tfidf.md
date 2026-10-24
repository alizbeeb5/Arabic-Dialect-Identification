# Task 14: transcript-only TF-IDF baseline

EEC-18 / W1-14. Fit a word/character TF-IDF union and multinomial logistic
regression on the verified Task 13 training transcripts. Jais is not rerun and
its dialect tag is never an input. This is a small in-source pilot baseline,
not a full-data or official-test result.

## Fixed experiment

Configuration: `configs/transcript_tfidf.json`, fixed before development scoring.
No hyperparameter search or selection on development results is performed.

- Word TF-IDF: unigrams/bigrams, Unicode word tokens, no lowercasing.
- Character TF-IDF: `char_wb`, lengths 3-5, no lowercasing.
- Both use min_df=1, sublinear TF and the vectorizer's L2 normalization.
- Logistic regression: lbfgs, C=1, tol=1e-4, maximum 1,000 iterations, seed 490,
  no class reweighting. Convergence warnings fail training rather than publish
  an unconverged result. Linear algebra is limited to one thread.

Both vocabularies and IDF weights fit only accepted training transcripts.
Development uses `transform` and `predict_proba` only. The checkpoint stores the
vectorizer, classifier, canonical labels, training-class prior, configuration
and training-input hashes. It contains no custom Python preprocessing class.
Native sklearn probability columns are explicitly reordered to the project's
canonical `configs/labels.json` order. Sigmoid/calibration claims are not made;
these are categorical multinomial probabilities, not independently calibrated.

## Text availability policy, fixed before fitting

Apply the same rule to each split: nonempty parsed text with status `ok` or
`truncated` is available if not refused and free of generated tag markers.
Known partial text is used as available text, overriding Task 13's conservative
`usable_for_text_model=false` on truncated rows. Unknown truncation is treated
as unavailable. Do not append a target, native tag, predicted dialect, source
name, file path or metadata to the input string.

Training: exclude unavailable text from fitting, record IDs/status/counts, and
require every canonical class still has support. In this release that leaves
492 training transcripts and 8 explicitly documented refusals. Unsupported
native Jais dialect tags do not invalidate otherwise usable text.

Prediction: keep every expected ID. For unavailable text, use the label prior
from the retained training rows, not development labels. Ties select the lowest
canonical index. Save prediction diagnostics separating text predictions from
prior fallbacks. Coverage of this fallback-capable system is not the same as
transcript availability. In this dev release, 196 clean + 4 partial texts should
all receive text predictions, with no fallback. Keep the full 200 denominator.

## Reproduce

Use a separate CPU environment with `environments/text-pilot.txt`; it records
the direct packages used here, not a complete transitive lock. From repo root,
set TRAIN_EXPORT and DEV_EXPORT to the original Task 13 export directories.
Set MODEL_RUN and DEV_RUN to new directories outside ordinary Git history.

```bash
python src/models/transcript_tfidf.py train \
  --export-dir "$TRAIN_EXPORT" \
  --manifest data/manifests/train_pilot.jsonl \
  --metadata data/manifests/train_pilot.metadata.json \
  --split train --config configs/transcript_tfidf.json \
  --output-dir "$MODEL_RUN"

python src/models/transcript_tfidf.py predict \
  --model "$MODEL_RUN/model.joblib" --export-dir "$DEV_EXPORT" \
  --manifest data/manifests/dev_pilot.jsonl \
  --metadata data/manifests/dev_pilot.metadata.json \
  --split dev --output-dir "$DEV_RUN"

python src/evaluation/evaluate.py \
  --manifest data/manifests/dev_pilot.jsonl \
  --metadata data/manifests/dev_pilot.metadata.json \
  --predictions "$DEV_RUN/predictions.jsonl" \
  --labels configs/labels.json --split dev \
  --model-name transcript-tfidf-logreg --output-dir "$DEV_RUN/evaluation"
```

Inputs are checked against export hashes and the frozen reference manifests;
text rows with unexpected label/source columns or targets disagreeing with the
references are rejected. Generated model files and prediction artifacts stay
outside Git; commit only implementation, configuration and reviewed summaries.

Outputs: `model.joblib`, config, training diagnostics, full prediction JSONL,
per-ID prediction-source diagnostics, metrics and provenance/environment record.
The common evaluator adds its standard confusion CSV and metric provenance.
Reloading the checkpoint must reproduce dev probabilities/predictions; use a
new output directory for that check. Do not load development text during fitting.

Tests:

```bash
python -m unittest discover -s tests -p 'test_transcript_tfidf.py' -v
```
