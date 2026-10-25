# Prediction evidence for the Week 1 comparison

These files contain only model predictions, class scores and provenance; no raw transcripts,
audio or trained checkpoints are committed. The original MFCC/acoustic prediction files remain
in their existing report folders.

- `jais_predictions.jsonl`: byte-identical to `dev_run/predictions.jsonl` in the user-supplied
  `task13_evidence.zip` (archive SHA256
  `d4c0808ed6ca286d496f224f49386ea55a995bce6c41ebdd11d90e272314b1bd`). Prediction SHA256:
  `daaa9ec6513ac62bfd401517e16f29f5d575c752e48b2fd4040c8959c254f422`.
- `text_predictions.jsonl`: regenerated from the original train/dev transcript exports using
  the unchanged Task 14 configuration and `environments/text-pilot.txt`. The resulting checkpoint
  SHA256 is `2c82c5c32574d6b58905042d96b59725389aa93bfc3733bb50f3c93471d866f4`;
  predictions SHA256 is `977e367b92b5235af17ede75ae170c8cb166ec205588f8ec8410f6b97cd6f08e`.
  Both match the original Task 14 release. `text_replay_record.json` records source, dependency,
  config and input identities from the replay at commit `c03bfad27c0c999dbfd237f8034b3d3e1d6b1109`.

The four saved prediction releases can now be analyzed without private run directories:

```bash
python src/analysis/week1_results.py --output-dir /tmp/week1-analysis-review
```

To repeat the artifact check and text refit, obtain the original evidence ZIP from team storage,
verify its hash, and extract it outside Git. It contains `train_export/`, `dev_export/`,
`train_run/` and `dev_run/`. Set `TRAIN_EXPORT` and `DEV_EXPORT` to those export directories;
`MODEL_RUN` and `DEV_RUN` must be new directories outside Git. From the repository root:

```bash
python tests/check_jais_pilot_evidence.py /path/to/task13_evidence.zip

python src/models/transcript_tfidf.py train \
  --export-dir "$TRAIN_EXPORT" \
  --manifest data/manifests/train_pilot.jsonl \
  --metadata data/manifests/train_pilot.metadata.json \
  --split train --config configs/transcript_tfidf.json --output-dir "$MODEL_RUN"

python src/models/transcript_tfidf.py predict \
  --model "$MODEL_RUN/model.joblib" --export-dir "$DEV_EXPORT" \
  --manifest data/manifests/dev_pilot.jsonl \
  --metadata data/manifests/dev_pilot.metadata.json \
  --split dev --output-dir "$DEV_RUN"

sha256sum "$MODEL_RUN/model.joblib" "$DEV_RUN/predictions.jsonl"
```

Do not rerun Jais for this check, fit text on dev exports, or tune parameters to match a score.
The common evaluator re-scores the frozen releases in the analysis command. The original
text-run timing remains historical; the replay is used for output identity, not a replacement
resource measurement. This is an artifact/text refit check, not an independent teammate's
full speech-model reproduction under Task 17.
