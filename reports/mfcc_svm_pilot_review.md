# W1-11 MFCC + SVM pilot sanity baseline: result review

Linear: [EEC-15 / W1-11](https://linear.app/eece490/issue/EEC-15/w1-11-run-the-mfcc-plus-svm-sanity-baseline).
Run on 2026-10-24 in a Colab CPU runtime. Method: [docs/mfcc_svm_baseline.md](../docs/mfcc_svm_baseline.md).

> **Result role: pilot sanity reference.** This checks the complete data → features →
> model → shared-evaluator path. It is not a competitive substitute for the learned
> acoustic or Jais baselines, and the development pilot is in-source ADI-20, not
> out-of-domain.

## Headline (frozen dev pilot, `w1-07-pilot-v1-e61cd569ac7b`)

| Development metric | Result |
|---|---:|
| Clips / dialect classes | 200 / 20 (10 per class) |
| Correct predictions | 49 / 200 |
| Accuracy | **24.50%** |
| Macro-F1 (all 20 supported classes) | **0.2383** |
| Coverage / evaluation mode | 100% / complete |
| Clip failures (train / dev) | 0 / 0 |
| Chance level (uniform guess) | 5% accuracy |
| Model revision | `mfcc-svm-pilot-sanity:35e8dac7384735e9` |

Model selection used only the 500-clip training pilot (5-fold stratified CV, seed 490):
C = 1 macro-F1 0.249 ± 0.045, C = 10 macro-F1 0.236 ± 0.052, so C = 1 was kept. The
margin is within CV noise. The dev result (0.238) agrees with the CV estimate, which is
consistent with no development-label leakage. Training resubstitution accuracy is
77.8%, so the RBF SVM partly memorises 25 clips per class; the dev and CV figures are
the ones to report.

## Per-dialect results (dev, 10 clips each)

| Dialect | Precision | Recall | F1 |
|---|---:|---:|---:|
| IRA | 0.89 | 0.80 | 0.84 |
| PAL | 0.75 | 0.60 | 0.67 |
| MAU | 0.71 | 0.50 | 0.59 |
| BAH | 0.26 | 0.60 | 0.36 |
| KSA | 0.38 | 0.30 | 0.33 |
| MSA | 0.24 | 0.50 | 0.32 |
| MOR | 0.27 | 0.30 | 0.29 |
| TUN | 0.25 | 0.20 | 0.22 |
| LIB | 0.25 | 0.20 | 0.22 |
| SYR | 0.22 | 0.20 | 0.21 |
| JOR | 0.25 | 0.10 | 0.14 |
| QAT | 0.10 | 0.20 | 0.13 |
| ALG, EGY, LEB, YEM | 0.11–0.12 | 0.10 | 0.11 |
| KUW, OMA, SUD, UAE | 0.00 | 0.00 | 0.00 |

With 10 clips per class, one clip moves a recall by 0.10; per-class differences are
indicative only.

## Error patterns

- **Three "attractor" classes absorb many errors.** BAH is predicted 23 times, MSA 21
  and QAT 20, against 10 true clips each. Their recall is inflated (BAH 0.60, MSA 0.50)
  while their precision is low (0.24–0.26; QAT 0.10). Clip-level MFCC averages appear
  to place many recordings near these classes regardless of dialect.
- **Gulf dialects are not mainly confused with each other.** Of 49 errors on true
  BAH/KSA/KUW/OMA/QAT/UAE clips, only 19 went to another Gulf label; the rest were
  spread across the inventory. Their low scores look like a lack of signal in these
  features rather than a fine-grained Gulf confusion.
- **Largest single confusions (4 clips each):** TUN → MSA, SYR → OMA, EGY → QAT.
- **Clearly separated classes:** IRA (8/10, precision 0.89), PAL (6/10, 0.75), MAU
  (5/10, 0.71). Whether this reflects dialect or recording-channel characteristics
  cannot be told from this pilot: source and speaker independence are unverified
  (see `split_audit.md`).

## Resources (CPU, from `run_record.json`)

| Stage | Seconds |
|---|---:|
| Training feature extraction (500 clips) | 12.77 |
| Fit including CV | 1.12 |
| Dev feature extraction (200 clips) | 3.53 |
| Dev prediction | 0.07 |
| **Wall time** | **17.74** |

Peak resident memory: 428.8 MiB. Audio was read from the Colab local disk; reading from
Drive would add I/O time. For context, the Jais pilot needed 486.47 s wall time and
11.85 GiB GPU memory on a T4 (reports/jais_pilot_review.md).

## Reproducibility and provenance

- Run twice in the same session: step-by-step in the notebook, then end-to-end with
  `src/experiments/mfcc_svm_baseline.py`. Both gave 49/200 correct.
- Audio: the 700 pilot clips were fetched with `src/data/fetch_pilot_audio.py` from
  `UBC-NLP/NADI_2026_ADI20_micro@c8da454c9f474bb075d3fe608940ee29e8ccfa43`; every clip
  was checked against the committed manifest's clip ID, byte length and SHA256.
- Code: the Colab checkout was `main` at `427bf38` with this branch's files unpacked on
  top, so the record shows `dirty: true`. The executed script and config hashes
  (`8f3c5629…`, `8a6dfc76…`) equal the original files at commit `67fe0d230d2e91c862c8cc7b32fa05f00860e785`; the evaluator
  hash (`857d2f29…`) equals `src/evaluation/evaluate.py` on `main`.
- Environment: Python 3.13.16, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1,
  soundfile 0.14.0, joblib 1.6.0.

Command (Colab):

```bash
python -u src/experiments/mfcc_svm_baseline.py \
  --audio-root /content/Arabic-Dialect-Identification-data/audio/adi20_micro/c8da454c9f474bb075d3fe608940ee29e8ccfa43 \
  --output-dir /content/Arabic-Dialect-Identification-data/runs/mfcc_svm_w1_11/20261024T212842Z-full
```

## Committed evidence (`reports/mfcc_svm_pilot/`)

`metrics.json`, `confusion_matrix.csv` (shared evaluator), `predictions.jsonl`,
`run_record.json`, `config.json`. The model (`model.joblib`, 0.5 MB), features and
decision scores are kept out of Git; the run record lists their SHA256 hashes. All
artifact hashes in the run record were re-verified against the downloaded run folder.

## Acceptance checks

- [x] Scaler and classifier use no development labels during fitting (training-only
      CV and refit; dev labels stripped before feature extraction; unit test permutes
      dev labels and gets identical predictions).
- [x] All 200 pilot clip IDs are represented (complete evaluation, 0 missing) and the
      result was rerun with an identical outcome.
- [x] Labelled a pilot sanity reference, not a competitive substitute for strong
      baselines.


## Post-review fixes and evidence limits

Review fixes add per-WAV SHA256/size verification to the experiment, preserve
integrity fields when removing development labels, reject unsupported window
settings, and pin notebook setup to the reviewed implementation. The original
run record and predictions above remain unchanged: their script hash identifies
the pre-fix source at `67fe0d230d2e91c862c8cc7b32fa05f00860e785`.

The committed predictions were independently re-scored: 49/200, accuracy 0.245,
macro-F1 0.23830908131117856, complete coverage. Available source/config/prediction
hashes match the historical record. Synthetic checks validate the fixes; no new
700-clip audio run was performed during this review. The new hash checks add I/O
cost that is not part of the historical 17.741-second measurement.

The actual model, feature array and diagnostic decision-score hashes require
the saved run ZIP for independent replay. They are not independently verified
from the committed predictions alone. Preserve or attach that run artifact
outside ordinary Git history for the experiment handoff.
