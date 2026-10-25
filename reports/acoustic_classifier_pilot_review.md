# W1-12 frozen-encoder acoustic classifier: pilot result review

Linear: [EEC-16 / W1-12](https://linear.app/eece490/issue/EEC-16/w1-12-train-and-evaluate-the-frozen-encoder-acoustic-classifier).
Run on 2026-10-25 in Colab on a Tesla T4. Method: [docs/acoustic_classifier.md](../docs/acoustic_classifier.md).
Artifacts: [acoustic_classifier_pilot/](acoustic_classifier_pilot/). Review by a different teammate pending.

> **Result role:** pilot reference on the frozen split `w1-07-pilot-v1-e61cd569ac7b` and the matched
> classifier reference for the verifier experiment. The development pilot is in-source ADI-20, not
> out-of-domain. The head is under-trained (below), so this is not the attainable accuracy of the encoder.

## Headline (dev pilot, 200 clips, 10 per dialect)

| Model | Correct | Accuracy | Macro-F1 | Training accuracy |
|---|---:|---:|---:|---:|
| MFCC + SVM sanity baseline (W1-11) | 49 / 200 | 24.50% | 0.2383 | 77.8% (resubstitution) |
| Transcript TF-IDF (W1-14) | 50 / 200 | 25.00% | 0.2364 | 98.4% |
| **Frozen AraBEST-RQ + linear head (this run)** | **67 / 200** | **33.50%** | **0.3149** | **53.8%** (epoch 30) |
| Jais direct dialect tag (W1-13) | 143 / 200 | 71.50% | 0.7046 | n/a |

Model revision `arabest-rq-frozen-linear-pilot:cc2984ca4936e6d2`. Coverage 100%, 0 clip failures in
train or dev, evaluation mode complete. Chance is 5% accuracy and log(20) = 2.996 loss.

Uncertainty: Wilson 95% interval for accuracy 27.3%–40.3%; one clip moves accuracy by 0.5 points
and a per-dialect recall by 0.10.

**Paired with MFCC + SVM on the same 200 clips:** both correct 32, this model only 35, MFCC only 17,
neither 116. Exact McNemar two-sided p = 0.018; paired bootstrap (10,000 resamples, seed 490) 95%
interval for the accuracy gain +2 to +16 points. The gain is unlikely to be noise on this pilot, but
it is one comparison on 200 in-source clips. Paired comparisons with the text models need their
per-clip predictions, which are not committed.

## Acceptance checks

| Check | Evidence (from `run_record.json`) |
|---|---|
| Encoder actually frozen, revision recorded | 0 trainable of 307,124,016 parameters, eval mode; SHA256 over all parameters and buffers identical before and after extraction (`0fe20c2f…`); revision `dff453e2b5f72f76211a523b1050691967e5a59b` with both checkpoint file hashes |
| No inconsistent feature recomputation | All 700 clips extracted once, in one run (`extraction_run_record.json`), batch size 1, one cache (`feature_config_sha256 e4b3c287…`). Both reported runs reused all 700 and extracted 0; train/dev cache-entry hashes (`e3534100…` / `03518ee5…`) and provenance hash are identical in all three runs |
| One documented command regenerates the predictions | Two separate `MODE = 'full'` runs (`20261025T102216142117Z`, `20261025T105406250501Z`) wrote byte-identical `checkpoint.pt`, `predictions.jsonl` (`3ef4b1fc…`), learning curve and confusion matrix; only evaluator timestamps differ. The committed predictions re-evaluate locally to the same metrics |

The 307,124,016 trainable-capable parameters plus two fixed positional-encoding buffers
(2,120,424 values) give the 309.24M checkpoint total in `acoustic_features.md`.

## Overfit gate (first)

40 training clips (2 per dialect), 300 full-batch updates at learning rate 1e-3: loss 2.9895 → 0.0606
(2% of the start), accuracy 5% → 100%. Passed. The starting loss sits at chance, so the features and
initialisation are well scaled; reaching 100% rules out flat loss, misjoined labels and degenerate
features before the longer run. The gate is necessary, not sufficient: 40 points in 848 dimensions
are separable for almost any features.

Feature statistics (final layer): mean |x| 0.100 train / 0.101 dev, per-dimension std 0.098 / 0.095,
no constant dimensions, no duplicate clips, mean pairwise cosine 0.51 / 0.54 (a shared direction,
not collapse). Train and dev distributions agree.

## Train and development behaviour

| Epoch | Updates | Train loss | Train acc | Dev loss | Dev acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 3.017 | 4.8% | 3.039 | 3.0% |
| 5 | 10 | 2.835 | 21.8% | 2.913 | 12.5% |
| 10 | 20 | 2.685 | 34.8% | 2.817 | 17.0% |
| 15 | 30 | 2.552 | 44.2% | 2.734 | 24.0% |
| 20 | 40 | 2.433 | 47.8% | 2.662 | 26.5% |
| 25 | 50 | 2.327 | 51.8% | 2.601 | 30.5% |
| 30 | 60 | 2.231 | 53.8% | 2.548 | 33.5% |

The head is **under-trained, not overfitting**. Training loss was still falling (−0.095 over the last
five epochs) and is far above zero; dev loss fell at every epoch and was lowest at the last.
Gradient norms stayed at 0.27–0.43, below the clip threshold of 1.0, so clipping never acted. The
cause is the update budget: batch 256 for 30 epochs is only 60 AdamW updates on 500 clips, while
the gate needed 300 updates to memorise 40. The same configuration gives about 7,900 updates on the
67,015-clip full training manifest, so the budget is a pilot-size limit. This diagnosis uses training-set
evidence; the dev curve is reported but did not change the configuration.

Confidence is low: the mean top softmax probability is 0.135 (0.175 when correct, 0.115 when
wrong). No calibration is claimed.

## Per-dialect results (dev, 10 clips each)

| Dialect | Precision | Recall | F1 | Predicted |
|---|---:|---:|---:|---:|
| MAU | 0.67 | 0.80 | 0.73 | 12 |
| BAH | 0.67 | 0.60 | 0.63 | 9 |
| IRA | 0.50 | 0.80 | 0.62 | 16 |
| PAL | 0.54 | 0.70 | 0.61 | 13 |
| JOR | 0.60 | 0.60 | 0.60 | 10 |
| KSA | 0.40 | 0.40 | 0.40 | 10 |
| MSA | 0.33 | 0.50 | 0.40 | 15 |
| SUD | 0.60 | 0.30 | 0.40 | 5 |
| MOR | 0.27 | 0.40 | 0.32 | 15 |
| YEM | 0.20 | 0.40 | 0.27 | 20 |
| ALG | 0.29 | 0.20 | 0.24 | 7 |
| LEB | 0.25 | 0.20 | 0.22 | 8 |
| QAT | 0.20 | 0.20 | 0.20 | 10 |
| EGY | 0.33 | 0.10 | 0.15 | 3 |
| KUW | 0.33 | 0.10 | 0.15 | 3 |
| LIB | 0.25 | 0.10 | 0.14 | 4 |
| TUN | 0.09 | 0.20 | 0.12 | 22 |
| OMA | 0.09 | 0.10 | 0.10 | 11 |
| SYR | 0.00 | 0.00 | 0.00 | 6 |
| UAE | 0.00 | 0.00 | 0.00 | 1 |

All 20 dialects are predicted at least once; predictions are not collapsed onto a few classes, but
TUN (22) and YEM (20) are over-predicted. SYR and UAE have 0/10 correct. Most frequent confusions
(3–4 clips each): QAT→TUN, ALG→TUN, LIB→TUN, MOR→TUN, YEM→QAT, TUN→ALG, UAE→KSA, MSA→IRA,
LEB→MAU, EGY→OMA. Several are geographic neighbours (Maghreb, Gulf), but with 10 clips per class
these are observations, not established error patterns.

## Full class-score order

The correct dialect is ranked first for 33.5% of clips, in the top 3 for 50.0% and in the top 5 for
64.0% (mean reciprocal rank 0.479, median rank 3.5). This matters for the verifier, which scores
(clip, candidate dialect) pairs: the reference ranking already places the correct label near the top
for most clips.

## Clip duration (observational)

Dev accuracy by duration: under 5 s 33.9% (59 clips), 5–10 s 21.1% (76), 10–20 s 38.3% (47),
20 s and longer 72.2% (18). Longer clips may carry more dialect evidence, but the groups are small,
not balanced by dialect, and not controlled; no causal claim follows.

## Resources (Colab Tesla T4)

| Step | Measurement |
|---|---|
| Feature extraction, 700 clips (112.4 min audio), batch 1 | 310.3 s end to end, 21.7 audio-s/s; about 61 s of it is encoding (500 train in 41.5 s, 200 dev in 19.5 s), the rest WAV hash checks and reads from Drive |
| Peak GPU memory during extraction | 12.15 GiB allocated (the 142 s dev clip) |
| Full run from the cache | 19.7 s wall, of which encoder load 13.7 s; overfit gate 0.49 s; training 0.15 s on CPU |
| Peak GPU memory, full run | 1.17 GiB (encoder weights only; nothing encoded) |
| Peak host memory | 3.4 GiB (extraction), 3.1 GiB (full run) |

A first attempt with extraction batch 8 extracted the 500 training clips but ran out of memory on
the dev pilot (one 6.0 GiB attention tensor for 8 clips padded to 142 s). That cache was deleted and
all 700 clips were re-extracted at batch 1, so no reported feature comes from the batch-8 attempt.
During the batch-1 run the allocator logged one failed 386 MiB request on the 142 s clip, freed its
cache and retried successfully; the run reported 0 failures. A 212 s clip from the full dev release
is unlikely to fit on a T4.

## Provenance

Code: base commit `d535541` plus uncommitted W1-12 files, uploaded as a zip (`aa7d0a7f…`); script
`119dd763…`, config `6d7df173…`, encoder code `dd9f8cf0…` (all in `run_record.json` and matching
the original PR source at `cc0fe5e6efea9cded41ba820391d1bcda159b434`). Environment: Python 3.13.16, torch 2.14.1, torchaudio 2.11.0+cu130,
speechbrain 1.1.2, numpy 2.5.4. The checkpoint (`cc2984ca…`, 71,677 bytes) is git-ignored and stays
in the run folder on Drive (`runs/acoustic_classifier_w1_12/20261025T102216142117Z_full/`).

## Limitations

- 200 in-source dev clips: wide intervals, no out-of-domain or full-data claim.
- Under-trained head under a fixed pilot budget; 33.5% is not the encoder's ceiling.
- Speaker and source independence of the pilot split are not verified; AraBEST-RQ was pretrained on
  Arabic YouTube speech and ADI-20 is YouTube-derived, and their overlap is not audited.
- Only the final encoder layer was used.

## Next justified experiment

In order of cost, each motivated by training-set evidence above:

1. **Give the head enough updates.** Keep the head and features, raise the update budget (more
   epochs or a smaller batch) until training loss plateaus, chosen on training loss or a split of
   the training pilot, never on dev. Cheap: CPU seconds from the existing cache. Note for EEC-19:
   its learning-rate grid {1e-4, 3e-4, 1e-3} at 60 updates would under-train both arms on the
   pilot; the controlled comparison should run on the full release or with a pilot budget that
   reaches a training-loss plateau.
2. **Layer choice.** All 26 layers are already cached. Compare single layers or a learned layer
   mixture with the same head, selected without dev labels; intermediate SSL layers often carry
   more speaker and accent information than the final one.
3. **Encoder fine-tuning,** only if 1–2 plateau: unfreeze the top Conformer layers on the full
   training release with a GPU larger than a T4 and a decision on long clips. This is the first
   fine-tuning experiment the pilot justifies; fine-tuning now would confound an under-trained head
   with encoder adaptation.

## Post-review corrections

The extraction runner now isolates hash, decoding and model failures, retries failed batches
clip by clip, and preserves all dev IDs as explicit audio/inference failures. Cache-write and
provenance errors still stop execution. The notebook is configured to fetch an immutable source
commit by default. The 67/200 result and historical run records above are preserved; these
review corrections have offline regression coverage, without a new full-audio/GPU run.
