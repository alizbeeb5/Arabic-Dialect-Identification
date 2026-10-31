# W2-02 text-data scaling screen: 500 → 2,000 training clips

Linear: [EEC-24 / W2-02](https://linear.app/eece490/issue/EEC-24/w2-02-screen-text-data-scaling-from-500-to-2000-clips-on-colab).
Run on 2026-10-31 in one Colab T4 session. Notebook: [`notebooks/W2_02_Text_Scaling.ipynb`](../notebooks/W2_02_Text_Scaling.ipynb).

> **Result role: exploratory screen.** One expanded training size, one fixed classifier,
> 200 in-source ADI-20 development clips (10 per dialect). It tests whether training-text
> quantity contributes to the Week 1 text gap. It is not a scaling curve, a full-corpus
> result, a fusion result or an official-test comparison.

## Headline (frozen dev pilot `w1-07-pilot-v1-e61cd569ac7b`, identical 200 IDs)

| Training set | Usable training texts | Training accuracy (all IDs) | Dev correct | Dev accuracy | Dev macro-F1 |
|---|---:|---:|---:|---:|---:|
| Original pilot (500) | 492 | 98.4% | 50 / 200 | 25.0% | 0.2364 |
| Expanded nested set (2,000) | 1,976 | 98.1% | **75 / 200** | **37.5%** | **0.3601** |
| Change | ×4.0 | −0.3 pt | **+25** | **+12.5 pt** | **+0.1237** |

Paired on the same 200 clips: both correct 43, **only the 2,000 model correct 32**, only the
500 model correct 7, both wrong 118. Exact two-sided McNemar test on the 32/7 discordant
pairs: **p = 7.0 × 10⁻⁵**. Both arms predict all 200 dev clips from text (no prior fallback),
so prediction coverage is 100% in both.

For context only (same dev IDs, Week 1): MFCC + SVM 24.5%, frozen AraBEST-RQ + linear 33.5%,
Jais direct dialect tag 71.5%. The text branch remains far below Jais's own tag.

## What changed and what did not

**Only the quantity of training text changed.**

- **Training IDs.** `data/manifests/train_scaling_2000.jsonl`, 100 per dialect, split version
  `w2-02-text-scaling-v1-e2dbe4ee0b9a`. Produced by `src/data/expand_train_pilot.py`, which
  reruns the frozen W1-07 selector (`freeze_splits.py`, unchanged code SHA256 `806d41e6…`,
  seed 490, same audited full-release manifests and exclusion rules) with 100 instead of
  25 clips per label. The round-robin draw order does not depend on the quota, so the
  first 25 picks per label are exactly the original pilot. The script fails unless the
  2,000 IDs strictly contain the original 500, are training-only and balanced, and the 200
  dev IDs are unchanged. No development result was read. `tests/test_expand_train_pilot.py`
  regenerates all four manifest/metadata files byte for byte.
- **New transcriptions.** Only the 1,500 new IDs (`train_scaling_new_1500.jsonl`, 75 per
  dialect) were added. They used the same IFM/Jais-ASR revision `1323995f…`, prompt,
  float16, batch size 2, `max_new_tokens` 128 and adapter code as W1-13. The source was
  pinned at `e66ae0e2`, where the adapter, exporter and evaluator are byte-identical to the
  W1-13 commit `26b36593`. No generation-limit or prompt tuning was done.
- **Classifier.** `configs/transcript_tfidf.json` unchanged for both arms: word 1–2-grams +
  char_wb 3–5-grams, sublinear TF, logistic regression lbfgs, C = 1, max_iter 1,000,
  seed 490, no class weighting. Vocabulary, IDF and classifier are fitted only on retained
  training text. No hyperparameter or seed search.
- **Text policy (unchanged).** `ok` and known-partial `truncated` text is used. Refused text
  is excluded from fitting and reported. Unavailable dev text would receive the retained
  training class prior; none occurred. Dialect tags, labels and source metadata never
  enter text features (enforced by the existing schema check).

### Deviation: original transcripts re-generated, not copied

The W1-13 evidence archive (`task13_evidence.zip`) was not reachable from this run. The
original 500 training and 200 dev clips were therefore re-transcribed in the same session,
with the same adapter, revision, settings and package set as the 1,500 new clips. This
departs from the card's "reuse 500 training and 200 dev outputs" instruction. The
re-generated outputs are functionally identical to the Week 1 release:

| Check against the committed Week 1 evidence | Result |
|---|---|
| Jais direct dev labels vs `reports/week1_evidence/jais_predictions.jsonl` | 200 / 200 identical |
| Dev transcript statuses | 196 ok + 4 truncated (Week 1: 196 + 4) |
| Training refusals in the original 500 | 8 (Week 1: 8), 492 usable |
| 500-arm text dev labels vs `reports/week1_evidence/text_predictions.jsonl` | 200 / 200 identical |
| 500-arm dev probability vectors (all 200 × 20) | identical; maximum absolute difference 0.0 |
| 500-arm dev accuracy / macro-F1 | 50 / 200, 0.23642323859715164 (exact Week 1 value) |

The checkpoint and prediction-file SHA256 differ from Week 1 (`cafd34a8…` vs `2c82c5c3…`).
This is because the serialized bundle and every prediction row embed provenance that
necessarily changes: input export summary hashes (run IDs and timings) and the checkpoint
hash itself as `model_revision`. Python was also 3.13 here vs 3.12 in the replay; the
scientific packages were identical (numpy 2.3.5, scipy 1.17.0, scikit-learn 1.8.0). Byte
identity of the transcript text itself was not checked, because the archive was
unavailable. Because all three sets of transcripts come from one environment, the 500 vs
2,000 comparison is internally consistent.

## Availability and transcription accounting

| Jais run | Clips | Usable text (ok + truncated) | Refused | Truncated | Dialect-tag failures | Wall time | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke (first new clip per dialect) | 20 | 20 | 0 | 0 | 0 | 3.1 min* | 10.9 GiB |
| New 1,500 (20 smoke clips were cache hits) | 1,500 | 1,484 | 16 | 1 | 19 (16 refused, 3 unrecognised tag) | 30.4 min | 11.1 GiB |
| Original 500 (re-run) | 500 | 492 | 8 | 0 | 14 (8 refused, 6 unrecognised tag) | 11.3 min | 11.1 GiB |
| Original dev 200 (re-run) | 200 | 200 | 0 | 4 | 0 | 5.9 min | 11.9 GiB |

\*Includes the one-time model download. Wall times include model loading and audio/cache
I/O, as recorded by the adapter; GPU-only generation time is not separated.

The merged 2,000 export holds 1,975 ok + 1 truncated + 24 refused. All 24 refused IDs are
listed in `training_diagnostics.json` and excluded from fitting. Every dialect retains
94–100 training texts (OMA 94 is the minimum). Unrecognised dialect tags affect only
Jais's direct label, never the transcript. No clip was silently dropped: every run has
one record per expected ID.

**Measured incremental cost.** 1,480 newly generated outputs in 30.4 min wall time
(≈ 0.81 clips/s end-to-end on a T4), plus ≈ 9 min to fetch the 1,500 WAVs from the pinned
Hugging Face release. The Week 1 planning extrapolation was 51–61 min, so the measured
run took about half that. Text fitting took 7.0 s (2,000 arm) vs 1.4 s (500 arm) on CPU.
At the measured rate, transcribing the full 67,015-clip training split would take roughly
**23 T4-hours** excluding setup and retries. This is a planning extrapolation, not a measurement.

## Training fit

| | 500 arm | 2,000 arm |
|---|---:|---:|
| Fit matrix | 492 × 44,997 | 1,976 × 118,047 |
| Word / char features | 13,388 / 31,609 | 45,354 / 72,693 |
| lbfgs iterations | 11 | 18 |
| Training accuracy (all IDs, refused → prior) | 98.4% | 98.1% |

Both models fit their retained training text almost perfectly, so the train–dev gap remains
large (≈ 61 points at 2,000 vs ≈ 73 at 500). More data narrowed the gap but did not
remove the overfitting.

## Per-dialect behaviour (dev recall, 10 clips each)

| Dialect | 500 | 2,000 | Δ | | Dialect | 500 | 2,000 | Δ |
|---|---:|---:|---:|---|---|---:|---:|---:|
| MSA | 8 | 5 | **−3** | | KUW | 0 | 0 | 0 |
| BAH | 6 | 7 | +1 | | LEB | 2 | 3 | +1 |
| TUN | 3 | 8 | **+5** | | LIB | 0 | 2 | +2 |
| ALG | 2 | 2 | 0 | | MAU | 2 | 3 | +1 |
| EGY | 6 | 7 | +1 | | MOR | 7 | 8 | +1 |
| IRA | 3 | 6 | **+3** | | OMA | 0 | 1 | +1 |
| JOR | 2 | 3 | +1 | | PAL | 1 | 3 | +2 |
| KSA | 0 | 4 | **+4** | | QAT | 1 | 2 | +1 |
| SUD | 1 | 1 | 0 | | SYR | 3 | 3 | 0 |
| UAE | 2 | 4 | +2 | | YEM | 1 | 3 | +2 |

15 dialects improved, 4 were unchanged and 1 got worse. KSA, LIB and OMA moved off 0/10;
KUW stays at 0/10. **MSA is the only regression** (8 → 5; F1 0.76 → 0.56): its three lost
clips were predicted MAU, ALG and MAU. One plausible reading is that a larger dialectal
vocabulary dilutes the formal-register cues that made MSA easy at 500 clips, but this was
not tested. Per-class changes rest on 10 clips each and are indicative only.

**Complementarity with Jais.** Of Jais's 57 direct-tag errors on these dev clips, the
2,000-clip text model is correct on 22 (the 500-clip model on 16). This is evidence of
complementary signal only; no fusion was trained or claimed.

## Interpretation and recommendation

The hypothesis that the Week 1 text result was partly limited by training size is
**supported on this screen**. With a 4× larger nested training set and nothing else
changed, dev accuracy rose from 25.0% to 37.5% and macro-F1 from 0.236 to 0.360. The
paired gain is unlikely to be chance on these 200 clips (p ≈ 7 × 10⁻⁵). It is still
one size point on a 200-clip in-source pilot: it does not establish a scaling trend, a
ceiling, or how text would perform out of domain.

**Recommendation: request a separately scoped larger training-team experiment.** Do not
stop the text branch. A suggested scope for the team to decide:

1. Two or three further nested sizes (e.g. 5,000 and 10,000, or the full 67,015 split)
   with the same frozen classifier, to see whether the gain continues or saturates
   (≈ 23 T4-hours for full-split transcription at the measured rate).
2. Score on the full ADI-20 dev split, not only the 200-clip pilot, to tighten per-dialect
   estimates. Add Casablanca validation as a diagnostic only, consistent with W2-03.
3. Separately pre-register any regularisation or feature changes. This task deliberately
   held the classifier fixed, so it says nothing about them. MSA's regression should be
   checked there.

## Outputs and where they are

**Committed (this PR):**

- `src/data/expand_train_pilot.py`, `data/manifests/train_scaling_2000.jsonl` (+ metadata,
  SHA256 `41514213…7929`), `train_scaling_new_1500.jsonl` (+ metadata, `cf727f13…ce94`)
- `src/data/merge_text_exports.py`: combines the 500 and 1,500 exports, validated with the
  model's own loader, records copied unchanged
- `src/analysis/text_scaling_compare.py`: re-scores both dev prediction files with the
  shared evaluator, plus the paired/McNemar comparison
- `notebooks/W2_02_Text_Scaling.ipynb`: the exact Colab cells run
- `tests/test_expand_train_pilot.py`
- `reports/week2_text_scaling/`: `comparison.json`, both dev prediction files (all-20 class
  probabilities), dev metrics, prediction-source diagnostics and training diagnostics

**Kept outside Git (Drive, `MyDrive/W2-02-text-scaling/`):** WAVs, Jais cache, raw
transcripts, model checkpoints (`train_500/model.joblib` `cafd34a8…`, `train_2000/model.joblib`
`3788d30c…`), run records and logs. The full evidence bundle is `w2_02_evidence.zip`
(SHA256 `9c24d208…7cf0`); available on request.

## Reproduce

Manifests (CPU, any checkout of this branch):

```bash
python src/data/expand_train_pilot.py            # regenerates the four manifest/metadata files
python -m pytest tests/test_expand_train_pilot.py
```

Text comparison, given the three Jais exports (`TRAIN500`, `NEW1500`, `DEV200`) and
`environments/text-pilot.txt`:

```bash
python src/data/merge_text_exports.py --original-export "$TRAIN500" --new-export "$NEW1500" --output-dir "$OUT/train2000_export"
for arm in 500 2000; do
  if [ $arm = 500 ]; then EX="$TRAIN500"; MF=train_pilot; else EX="$OUT/train2000_export"; MF=train_scaling_2000; fi
  python src/models/transcript_tfidf.py train --export-dir "$EX" --manifest data/manifests/$MF.jsonl \
    --metadata data/manifests/$MF.metadata.json --split train --output-dir "$OUT/train_$arm"
  python src/models/transcript_tfidf.py predict --model "$OUT/train_$arm/model.joblib" --export-dir "$DEV200" \
    --manifest data/manifests/dev_pilot.jsonl --metadata data/manifests/dev_pilot.metadata.json --split dev --output-dir "$OUT/dev_$arm"
done
python src/analysis/text_scaling_compare.py --train-500 "$OUT/train_500" --dev-500 "$OUT/dev_500" \
  --train-2000 "$OUT/train_2000" --dev-2000 "$OUT/dev_2000" --output "$OUT/comparison.json"
```

Expected tolerance: identical dev labels and probabilities given the same exports. The
logistic-regression fit is deterministic (seed 490, single-threaded linear algebra).
Jais transcription is GPU-generated: the full notebook rerun reproduces it on a T4 with
the recorded package set.

## Acceptance status

- [x] Expanded IDs are training-only, nested and deterministic; dev IDs and targets unchanged.
- [x] Only data quantity changes in the primary comparison; all 24 unavailable training texts are accounted for; dev has none.
- [x] Training/source/config hashes recorded; clean-checkout commands above.
- [ ] **Different-teammate verification of saved predictions and evaluation: pending** (reviewer and commands to be recorded on EEC-24).
- [x] No full-corpus transcription, transformer fine-tuning, fusion or verifier is claimed.
- [ ] Deviation (re-transcribed original 700 instead of reusing the archive) to be acknowledged by the reviewer.
