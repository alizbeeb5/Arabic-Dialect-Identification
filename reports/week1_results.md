# Week 1: pilot comparison, error overlap and resource estimate

Task: [EEC-20 / W1-16](https://linear.app/eece490/issue/EEC-20/w1-16-combine-pilot-results-errors-and-measured-resource-costs).
Analysis date: 2026-10-25. Implementation complete; independent teammate review pending.
Evidence snapshot: repository `c03bfad27c0c999dbfd237f8034b3d3e1d6b1109` plus the exact-hash
Jais and text predictions added in [week1_evidence/](week1_evidence/).

## Decision supported by the pilot

Keep Jais direct tags as the strongest observed reference. First give the frozen acoustic
head enough optimization updates, then run the matched classifier-versus-verifier experiment.
The text branch has complementary successes but needs more training evidence before a fusion
claim. Increasing encoder size is not yet justified by these runs.

These are **in-source ADI-20 development results**, not official-test results. No score here is
directly comparable with Thakaa's official-test accuracy, and no competition win is claimed.

## Common evaluation and comparison

Every row below is independently re-scored from its saved predictions by the same evaluator.
Frozen split: `w1-07-pilot-v1-e61cd569ac7b`; dataset revision
`c8da454c9f474bb075d3fe608940ee29e8ccfa43`; dev has exactly 200 clips, ten per dialect.
Manifest SHA256: `fb8684c35ef68e25834d5a4c82cddbd31044a899a63b665c41d3f9a179052d02`.
All four ID sets match exactly; duplicate, missing and unexpected IDs reject the analysis.
The complete prediction hashes, exact F1 values and descriptive intervals are in
[week1_analysis/summary.json](week1_analysis/summary.json).

| Model / saved predictions | Split | Scored | Accuracy | Macro-F1 | Historical runtime and scope | Historical memory |
|---|---|---:|---:|---:|---|---|
| [MFCC + linear SVM](mfcc_svm_pilot/predictions.jsonl) | frozen pilot v1 | 200/200 | 24.50% (49) | 0.2383 | 17.741 s CPU end to end for 500 train + 200 dev; includes extraction and CV | 428.8 MiB host RSS; CPU only |
| [Frozen AraBEST-RQ + linear head](acoustic_classifier_pilot/predictions.jsonl) | frozen pilot v1 | 200/200 | 33.50% (67) | 0.3149 | 310.335 s fresh extraction for 700 clips; separate cached full run 19.671 s, including 13.684 s encoder load and 0.153 s head fit | Extraction: 12.15 GiB GPU / 3452.4 MiB host; cached run: 1.17 GiB GPU / 3159.8 MiB host |
| [Jais direct dialect tag](week1_evidence/jais_predictions.jsonl) | frozen pilot v1 | 200/200 | 71.50% (143) | 0.7046 | Train: 980.732 s for 500 rows (20 cache hits); dev: 486.473 s for 200 fresh rows; adapter load/I/O/generation included, final export excluded | Train/dev: 11.06/11.85 GiB allocated GPU; host RSS not recorded |
| [Transcript TF-IDF + logistic regression](week1_evidence/text_predictions.jsonl) | frozen pilot v1 | 200/200 | 25.00% (50) | 0.2364 | 0.6055 s CPU vocabulary + model fit on 492 usable training texts; **excludes Jais transcription**, imports, validation, prediction and serialization | CPU only; original host peak not measured |

Coverage is 100% for all four dev prediction releases. Runtime scopes differ and do not form a
fair inference-speed ranking. GPU measurements above are allocated memory, not reserved memory
or the full device requirement. Acoustic and Jais runs used a Colab Tesla T4; the text pilot ran
in a separate CPU environment.

| Model | Run / model reference | Configuration and detailed record |
|---|---|---|
| MFCC | `20261024T212842Z-full`; model SHA256 `35e8dac7384735e9faa29fabf9a721527b375066bec15e1d858fe6689d579ef4` | [config](mfcc_svm_pilot/config.json), [run record](mfcc_svm_pilot/run_record.json), [review](mfcc_svm_pilot_review.md) |
| Acoustic | `20261025T102216142117Z_full`; head SHA256 `cc2984ca4936e6d2b69a17b714045dbe26ea805093a04500cc8dbdd53ac5164c`; encoder revision `dff453e2b5f72f76211a523b1050691967e5a59b` | [config](../configs/acoustic_classifier.json), [run record](acoustic_classifier_pilot/run_record.json), [extraction record](acoustic_classifier_pilot/extraction_run_record.json), [review](acoustic_classifier_pilot_review.md) |
| Jais | `jais_task13/20261024T185108716428Z`; dev suffix `20261024T191346757507Z`; checkpoint revision `1323995fa189c479369a3929be0e95039eba5cba` | [metrics](jais_pilot_metrics.json), [method](../docs/jais_pilot.md), [review](jais_pilot_review.md) |
| Text | published source `b474536a35ee265fe4b092c3c8ce67a114d08af9`; model SHA256 `2c82c5c32574d6b58905042d96b59725389aa93bfc3733bb50f3c93471d866f4` | [config](../configs/transcript_tfidf.json), [metrics](transcript_tfidf_pilot_metrics.json), [replay record](week1_evidence/text_replay_record.json), [review](transcript_tfidf_pilot_review.md) |

The text release was refitted from the verified original Task 13 exports in the recorded CPU
dependency versions. Both the checkpoint hash and the complete dev prediction-file hash match
the historical release exactly. No Jais inference, acoustic extraction or acoustic fitting was
rerun for this analysis. Raw transcripts and checkpoints remain outside Git.

## Error overlap: is there useful complementary information?

Counts below join by clip ID, not row position. "A only" means A is correct and B is wrong.

| A / B | Both correct | A only | B only | Both wrong | Same wrong label |
|---|---:|---:|---:|---:|---:|
| Acoustic / Jais | 53 | 14 | 90 | 43 | 8 |
| Jais / Text | 34 | 109 | 16 | 41 | 2 |
| Acoustic / Text | 20 | 47 | 30 | 103 | 6 |
| MFCC / Acoustic | 32 | 17 | 35 | 116 | 21 |

The acoustic model fixes 14 of Jais's 57 errors; text fixes 16. Their successful corrections
overlap on eight clips: together they cover 22 Jais errors, and acoustic contributes six cases
that both Jais and text miss. All four models are wrong on 33 clips.

Those counts identify candidate complementary signals. A gold-label oracle choosing between
Jais and acoustic would score 157/200; Jais and text 159/200; Jais/acoustic/text 165/200.
These are **diagnostic upper bounds on this sample, not deployable systems or fusion results**.
In particular, choosing when to override a correct Jais output is still unsolved. Jais exports
hard tags, not verified 20-class probabilities; probability averaging with it is not available.

The complete paired counts are in [summary.json](week1_analysis/summary.json), and all 200
aligned predictions and audio locations are in [per_clip.csv](week1_analysis/per_clip.csv).

## Largest confusions and concrete examples

| Model | Largest directed confusions (true -> predicted) |
|---|---|
| Jais | BAH -> MSA: 6; TUN -> ALG: 6; KUW -> QAT: 5 |
| Acoustic | QAT -> TUN: 4; ALG -> TUN, EGY -> OMA, LEB -> MAU, LIB -> TUN: 3 each (other ties exist) |
| Text | ALG -> MOR, KSA -> ALG, KUW -> SYR, OMA -> KSA, PAL -> LEB: 3 each (other ties exist) |
| MFCC | EGY -> QAT, SYR -> OMA, TUN -> MSA: 4 each |

Examples below use the first 12 hex characters as readable identifiers. Full IDs and portable
audio paths are in [per_clip.csv](week1_analysis/per_clip.csv); additional examples from the largest
confusions are in [error_examples.csv](week1_analysis/error_examples.csv).
Selection is deterministic within the largest confusion groups, not a representative sample.

| Clip prefix / seconds | Gold | MFCC | Acoustic | Jais | Text | Observation |
|---|---|---|---|---|---|---|
| `0785ef3d1655` / 27.36 | BAH | BAH | BAH | MSA | BAH | Jais error corrected by all three other branches |
| `344b79595f89` / 4.90 | BAH | OMA | OMA | MSA | OMA | All wrong; agreement on OMA is not evidence that OMA is correct |
| `0005c2e8edbe` / 12.00 | TUN | ALG | JOR | ALG | LIB | All wrong, with different proposed alternatives |
| `0948e611991f` / 4.231 | TUN | TUN | ALG | ALG | EGY | MFCC supplies the only correct prediction here |
| `34895e6b17e5` / 9.088 | KUW | QAT | QAT | QAT | QAT | Shared KUW/QAT failure across every branch |
| `0d08b34476a2` / 11.904 | QAT | ALG | TUN | QAT | PAL | Overriding Jais with the acoustic prediction would hurt |
| `af9ebb851ae4` / 6.020 | TUN | MSA | TUN | ALG | UAE | Acoustic correct when both Jais and text are wrong |

The cached transcript for `0785ef3d1655` has formal institutional wording; `344b79595f89` has a
very short religious expression. This suggests a **hypothesis** that register or scarce lexical
evidence can disagree with a country label. It does not establish the spoken variety: these
transcripts are model outputs and the audio was not listened to in this review. Do not relabel
these examples or claim transcript correctness. Similarly, TUN/ALG and KUW/QAT confusions are
consistent with difficult regional distinctions, but audio/annotation review is needed to
separate that explanation from transcription, source or label issues.

## Implementation/data issues versus model limitations

| Verified observation | Interpretation and action |
|---|---|
| Acoustic overfit gate reaches 100% on 40 training clips; full head gets only 60 updates, 53.8% training accuracy and still-falling training loss | Evidence for an insufficient optimization budget. Reuse cached features; set a larger common update budget before testing head designs. This does not establish the encoder's accuracy ceiling. |
| Text fits all 492 retained texts correctly; dev is 25% | Large generalization gap on a very small lexical training set. More training text and training-only model selection are justified; a larger classifier alone is not. |
| Jais has 8 refused train rows and 6 unsupported native dialect tags; dev has 4 partial MSA transcripts | Explicit availability/coverage issues, not hidden row drops. Text fits 492 rows and retains all 200 dev IDs, including the four partial texts. All four partial MSA texts are classified correctly by text and acoustic; three by direct Jais. |
| Acoustic batch 8 failed on a long dev clip; batch 1 completed with 12.15 GiB peak and a retry | A resource/configuration failure, not dialect confusion. Full dev includes a 212.38 s clip, beyond the pilot's 142.05 s maximum. Set a long-clip execution policy before full extraction. |
| No known speaker IDs; recording IDs missing for most clips; BAH pilot uses four known recording groups | Speaker/source independence is not established. Ten BAH clips are not ten independent recordings. Add separate-source evaluation and preserve the split audit limitations. |
| MFCC/acoustic reviews fixed cache, failure-handling or notebook source issues after the measured runs | Original scores and timing remain historical. Offline checks do not establish that the corrected implementation has completed a new full audio run. |

See [split audit](split_audit.md) and the individual reviews for exact provenance limits. One dev
clip changes overall accuracy by 0.5 percentage points and class recall by ten points. Descriptive
clip-level Wilson intervals are 19.1-30.9% (MFCC), 27.3-40.3% (acoustic), 64.9-77.3% (Jais), and
19.5-31.4% (text); recording clusters limit their population interpretation. A 200-clip pilot
does not establish a confident competitive ranking or out-of-domain performance.

## Compute and storage estimate

Full release: 67,015 train + 10,806 dev clips = 77,821 clips and 188.798 audio hours.
Calculations are regenerated in [summary.json](week1_analysis/summary.json).

| Work | Estimate on the measured T4 setup | Assumptions / limits |
|---|---|---|
| Frozen acoustic extraction, all clips | 8.69 h nominal; **8.69-17.38 occupied GPU hours** planning range | 6,741.802 pilot audio seconds / 310.335 extraction seconds = 21.724 audio-s/s. Duration-scaled full audio; 1-2x explicit planning buffer, not a statistical interval. Excludes cold download/startup and requires resolving long-clip memory. |
| Jais, fully cold | **44.17-52.58 occupied GPU hours** | 2.043-2.432 wall seconds per new output from train/dev logs. Train includes cache-hit overhead; dev is cold. This is an observed-rate bracket, not a guaranteed bound. Full lengths, failures, batching and filesystem speed can change it. |
| Jais, reuse the 700 pilot outputs | **43.77-52.11 hours** for 77,121 remaining clips | Cached train failures must be explicitly retained or retried under a versioned policy. Same checkpoint, prompt, generation and audio identities required for cache reuse. |
| Additional head/text experiments from caches | No defensible full-data timing yet | Pilot CPU fits are fast; linear extrapolation is unreliable for changed feature dimensions, epochs or solver behavior. Benchmark a full-data fit rather than promise a runtime. |

The external team provides A100/H100 resources, but no agreed hourly price or measured
A100/H100 throughput is recorded. **Dollar cost is therefore pending:** with a matching T4 rate
`p` per occupied GPU-hour, extraction costs `8.69p-17.38p` and cold Jais `44.17p-52.58p`, excluding
storage, setup and retries. These are formulas, not agreed dollar amounts. Do not substitute an
A100 price into T4 timings or divide by the reported cluster GPU count; parallel scaling and
the actual allocation are unmeasured. No paid job is launched by this task.

Measured preparation storage is 21.769 GB raw shards + 21.824 GB extracted WAV = **43.593 GB**
before caches, downloads and outputs (decimal GB). Dense float32 pooled acoustic payloads would
add 6.863 GB for 26 x 848 values per clip, or 0.264 GB for final-layer-only features. Archive
headers, provenance and filesystem overhead are additional; these payloads are calculated, not
measured full-cache sizes. Reserve at least 50.456 GB for raw + WAV + all-layer payload alone,
then add model downloads, transcript caches, outputs and headroom. Full transcript-cache size
and billed storage are unmeasured.

Compute features and transcripts once for each pinned identity; subsequent head/text fits reuse
them. The text branch's 0.6055 s fit is incremental cost after transcription, not an end-to-end
speech classification cost. Preserve raw generation diagnostics separately from parsed text,
targets and prediction tags to prevent target leakage. Coordinate one cache writer or safe
per-worker locations on the external team's storage.

## Proposed next experiments (for the Week 2 review)

1. **Optimization-controlled acoustic classifier versus verifier.** Expected failure mechanism:
   the current head is under-trained, so a comparison at 60 updates would confound optimization
   with objective design. Freeze the same cache, predeclare an update budget using training-only
   evidence, and give both arms matched selection budgets. Follow [the controlled specification](../docs/verifier_experiment.md),
   with any budget amendment recorded before running the comparison.
2. **Training-size test for the text branch, then a controlled combination test if warranted.**
   Expected failure mechanism: sparse lexical features memorize a tiny transcript set. Reuse
   cached texts; choose sizes/hyperparameters on training-only folds and keep the dev denominator.
   Complementary errors motivate investigation, but hard-tag Jais lacks calibrated class scores;
   a fusion rule must be specified and validated independently before claiming gains.
3. **Separate-source development evaluation and long-clip execution check.** Expected failure
   mechanisms: source/register cues inflate in-source performance, and long clips exceed memory.
   Audit/freeze Casablanca validation under the existing protocol and test the longest intended
   audio inputs before handing over a full extraction job. Do not tune on final test labels.

## Reproduce this analysis

From a checkout containing this report and its small prediction evidence files:

```bash
python src/analysis/week1_results.py --output-dir /tmp/week1-analysis-review
python -m unittest discover -s tests -p 'test_week1_results.py' -v
```

Use a new output directory. The command checks release hashes and common IDs, invokes the shared
evaluator, and writes `summary.json`, `per_clip.csv`, `per_dialect.csv` and `error_examples.csv`.
These outputs are deterministic and need no GPU, model checkpoint, raw audio or transcripts.

Original Jais evidence check and exact text-replay commands are documented in
[the evidence README](week1_evidence/README.md). This verifies exported artifacts and the text
refit; it does not count as a separate teammate's GPU/audio reproduction for Task 17.
