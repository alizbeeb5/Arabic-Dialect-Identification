# Task 14: transcript-only pilot result

Date: 2026-10-24. EEC-18 / W1-14. Implementation and CPU pilot complete;
review pending. Exact metrics: `transcript_tfidf_pilot_metrics.json`.

## Fixed first-run result

| Model | Dev correct / expected | Accuracy | Macro-F1 | Prediction coverage |
|---|---:|---:|---:|---:|
| Jais direct dialect tag (Task 13) | 143 / 200 | 71.50% | 0.7045528323025294 | 100% |
| Transcript TF-IDF + logistic regression | 50 / 200 | 25.00% | 0.23642323859715164 | 100% |

Both predictions use exactly the frozen `dev_pilot` IDs and all 20 dialects,
with ten dev references per class. This is an in-source development comparison;
no official-test, OOD or full-data claim follows from it. No word/char variants,
regularization search, seed selection or post-result tuning was performed.
The configuration was committed before scoring development.

## Data and feature isolation

Input: verified Task 13 exports from `task13_evidence.zip`, archive SHA256
`d4c0808ed6ca286d496f224f49386ea55a995bce6c41ebdd11d90e272314b1bd`.
Frozen split `w1-07-pilot-v1-e61cd569ac7b`. No Jais inference was rerun.

Fit data: 492 available training transcripts from the 500-clip pilot; the eight
refused rows are recorded and excluded from fitting. Every dialect retains
22-25 examples. Known partial text is accepted under the same rule in both
splits. Dev uses all 196 clean and four partial transcripts, with no prior
fallback. Missing/refused dev text would receive the class prior learned from
retained training rows, with a separate per-ID diagnostic and no row dropping.
The training evaluation retains all 500 IDs and therefore includes eight prior
fallbacks. Model prediction coverage is distinct from transcription coverage.

Only parsed transcript strings enter the two vectorizers. Gold targets are
joined by clip ID separately; labels, dialect tags and source fields are rejected
in text records. Both vocabulary/IDF fits use retained training text only;
development calls transform/predict only. Native sklearn score columns are
explicitly mapped into canonical ADI-20 order before prediction export.

Word unigrams/bigrams: 13,388 features. Character word-boundary 3-5-grams: 31,609.
Combined fit matrix: 492 by 44,997. Logistic regression uses lbfgs, C=1,
max_iter=1,000, tol=1e-4, no class weighting and seed 490. It converged in
11 iterations. CPU fit took 0.6055 seconds, measured over feature fitting and
classifier fitting with one linear-algebra thread; imports, input validation,
serialization and evaluation are excluded from that timing.

## Interpretation and failures

Full training-subset accuracy is 492/500 (98.4%). All 492 fitted texts are correct
in sample; the eight prior fallback predictions on refused rows are incorrect.
The much lower development accuracy is consistent with overfitting on this
small text dataset. This run does not identify whether data quantity, ASR
standardization, vocabulary choice or regularization is the dominant cause.
It does not establish the attainable performance of a larger text branch.

Dev recall is 80% for MSA, 70% for MOR and 60% for BAH/EGY. KSA, KUW, LIB and
OMA have 0/10 correct in this pilot. Ten examples per class limit these
class-level conclusions.

Paired dev correctness: both models correct on 34 clips; Jais alone correct on
109; text alone correct on 16; both wrong on 41. Those 16 complementary cases
justify examining disagreement in later work, but they do not establish a
working fusion improvement. Do not report an oracle combination as a model.

No dev clip or probability vector is missing. All 200 rows have finite,
normalized 20-way probabilities in the common label order. These are native
multinomial probabilities, not independently calibrated scores. C_avg remains
unavailable under the unverified official scoring setup.

## Reproducibility

Training source commit (local): `faa0084e61c4e68fa8fd030186121774e1e3eeed`, clean checkout.
The published source commit is `b474536a35ee265fe4b092c3c8ce67a114d08af9`.
Both have exactly the same Git tree:
`305d3a39192fe28a06d1a92942c570237b1846c0`. The run record preserves the
original local commit identity; the published commit supplies a fetchable source
reference without rewriting the experiment artifacts.
Checkpoint SHA256:
`2c82c5c32574d6b58905042d96b59725389aa93bfc3733bb50f3c93471d866f4`.
Recorded environment: Python 3.12.14, sklearn 1.8.0, NumPy 2.3.5, SciPy 1.17.0,
joblib 1.5.3 and threadpoolctl 3.6.0. CPU only; direct dependency pins are in
`environments/text-pilot.txt`.

A separate refit from the same cached transcripts, followed by checkpoint
reload and prediction, reproduced every complete prediction record exactly;
maximum absolute probability difference was 0.0. The common evaluator
independently reproduced accuracy/F1/coverage and emitted the standard confusion
CSV. Tests cover no dev vocabulary fitting, class-column ordering, prior
fallback/ties, partial-text policy, generated markers, missing classes,
serialization, training-on-dev rejection and export hash/schema isolation.

See `docs/transcript_tfidf.md` for exact training/prediction/evaluation commands.
The trained bundle contains the checkpoint, outputs, configuration, required
source/reference files and verified transcript inputs for local reproduction.
Raw transcripts and trained model files are kept outside ordinary Git history.
The existing recording/speaker independence limitations still apply.
