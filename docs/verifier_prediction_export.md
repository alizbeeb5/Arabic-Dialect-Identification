# Verifier prediction export

Companion to `verifier_experiment.md`; extends no existing schema. Follow
`data_contracts.md` and `configs/labels.json` exactly.

For each expected clip, compute one shared feature and all 20 raw logits in
canonical order. Select argmax; exact ties select the lowest canonical index.
Do not restrict candidates using known evaluation-country subsets. Export one
JSONL row per expected clip with `clip_id`, `predicted_label`, immutable
`model_revision`, `status: "ok"`, `scores` (20 finite floats), `score_labels`
(the canonical list) and `score_type: "logits"`.

The verifier's independent sigmoid scores need not sum to one and must NOT be
exported as `probabilities` under the current categorical schema. Raw logits
preserve ranking and satisfy the existing contract. A sigmoid is not evidence
of calibration; calibration requires a separately specified development-only
experiment, independent calibration/evaluation partitions, and a defined pair
prior. No threshold or abstention is used for closed-set identification here.

On decode failure write `status: "audio_error"`; on feature/scoring failure use
`status: "inference_error"`. Set `predicted_label: null`, include a short
`error_code`, and omit all score fields. Never invent zeros, a dialect label or
one-hot scores. The shared evaluator counts failures in its full denominator.
Reject duplicates, unknown IDs, missing rows, nonfinite scores and order errors.

Run the existing evaluator from repository root after prediction export:

```bash
python src/evaluation/evaluate.py \
  --manifest "$DEV_MANIFEST" \
  --metadata "$DEV_METADATA" \
  --predictions "$PREDICTIONS_JSONL" \
  --labels configs/labels.json --split dev \
  --model-name verifier-balanced-bce \
  --output-dir "$EVALUATION_DIR"
```

Use the classifier model name for the matching classifier run. These environment
variables must identify the actual frozen release and artifact, not illustrative
paths. Save the config, code commit, seeds, manifest/label/cache hashes,
checkpoint hash, selection results and command beside each prediction artifact.

For external benchmark files, join by clip ID to the frozen organizer row-order
manifest first. Write `logits.tsv` (20 tab-separated columns, no header) and
`predictions.tsv` (one zero-based ID per row, no header). Reject export if any
expected clip failed or lacks scores; never silently remove a row. This specifies
the repository's existing provisional export contract, not verified acceptance
by the official platform. C_avg remains unavailable until scorer and score
transform compatibility are verified; logits alone do not resolve that question.
