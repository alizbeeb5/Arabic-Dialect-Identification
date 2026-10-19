# Evaluation plan and data access
Version: 0.3 | Updated: 2026-10-19 | Owner: Ali Zbeeb
Task: [EEC-8 / W1-04](https://linear.app/eece490/issue/EEC-8/w1-04-resolve-official-evaluation-access-and-model-eligibility)

## Decision
Use a local, labeled cross-dataset evaluation for the eight-week research project. Test whether a dialect-verification head improves generalization over a standard classifier using the same speech encoder, training data, and evaluation clips.

**Initial target choice: Casablanca.** It is accessible and has a compatible country inventory. Select it for the first pilot; freeze the final evaluation release only after the source-overlap and checkpoint-exposure audit.

NADI 2026 remains a published reference. Its displayed evaluation phase ended July 29 and no active phase was shown in the user's October 19 screenshot. Its public test release lacks labels. We will not depend on organizer contact, manually label its 878 clips, or claim a win over Thakaa from another dataset.

## Data we will use
| Role | Data | Rule |
|---|---|---|
| Source training | ADI-20 / organizer ADI-20 Micro training split | Train all compared heads on the same release |
| In-source checks | Source validation split | Sanity checks; declare any use in model selection |
| Target development | Casablanca validation | Choose checkpoints and settings; no gradient updates in the primary experiment |
| Target final test | Casablanca test | Evaluate the frozen system after the audit |
| Optional adaptation | A separately budgeted target-training subset | Separate experiment; do not consume final-test clips |

Development labels are available for selection, so describe this as cross-dataset evaluation with target-domain development, not target-blind domain generalization. The 50-clip adaptation experiment is not part of the initial comparison.

## Why Casablanca
The official release has audio and validation/test partitions for eight country dialects. Its paper describes TV-drama episodes selected by native speakers and reports a dialect-identification experiment. The target labels come from the release's country configurations, not model-generated labels.

| Configuration | Our code | Validation clips | Test clips |
|---|---|---:|---:|
| Algeria | ALG | 844 | 843 |
| Egypt | EGY | 846 | 846 |
| Jordan | JOR | 848 | 848 |
| Mauritania | MAU | 953 | 953 |
| Morocco | MOR | 1045 | 1045 |
| Palestine | PAL | 667 | 667 |
| UAE | UAE | 813 | 813 |
| Yemen | YEM | 803 | 803 |
| **Total** | **8 classes** | **6819** | **6818** |

These are published release counts, before our audit or fixed inclusion rules. The public release is about 9.16 GB; it contains validation and test only, not all 48 hours described for the full research collection.

Access: [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca).
Pinned revision: 8951b1b88e28c1107142ced57967b8d16350951d.
Repository is public and ungated; declared license: CC BY-NC-ND 4.0. Cite the dataset, retain original files, and distribute code/manifests/results rather than repackaged audio.

Access checked: metadata and schema retrieved; one Algerian validation audio clip downloaded and read as PCM WAV (44.1 kHz, stereo, approximately 0.923 seconds). Full dataset ingestion, listening-based quality review, and test evaluation have not been performed.

## Rules for a fair comparison
1. Keep the 20-label output mapping. Casablanca supports eight reference classes; predictions to the other 12 still count as errors. Report accuracy and macro-F1 over the fixed eight supported classes, per-dialect results, failure rate, and resource cost.
2. Use identical clips, preprocessing policy, encoder initialization, training data, and selection budget for classifier-versus-verifier experiments. Report any additional model resources separately.
3. Keep Casablanca test untouched by training and tuning. Inspect validation for development; run the frozen test evaluation after split approval.
4. ADI-17 already contains YouTube speech. Casablanca being a separate collection is evidence of a dataset/domain change, not proof of zero shared sources. Audit recording IDs and audio duplicates against source training.
5. The inspected Casablanca row schema exposes audio, seg_id, transcription, gender, and duration, but no explicit episode or speaker ID. Do not assume a seg_id prefix is an episode. Recover grouping if possible and check validation/test overlap; otherwise disclose unverified episode/speaker independence.
6. Check each encoder's training and fine-tuning history. Jais-ASR already reports Casablanca ASR evaluation on its card; this is not proof of training contamination, but prevents claiming the benchmark was never encountered. Unknown pretraining overlap must be disclosed.
7. Dataset/config names establish targets and bookkeeping only. Audio is the predictive input; human transcripts, filenames, and country metadata do not enter model features or prompts.

## Secondary candidate
[ARCADE](https://huggingface.co/datasets/riotu-lab/ARCADE-full) offers radio recordings and country metadata across 19 countries. The release has 6907 annotation rows, while the paper reports 3790 unique audio segments. It needs deduplication, annotation consensus, keep/skip and MSA/dialect filtering, and checks that station country matches spoken dialect. It also needs a grouped split. Keep it as an optional second-domain evaluation, rather than delaying the Casablanca pilot.

ARCADE revision: 9bc8458288eb8290b9c676e3b1065803970581a0; public/ungated; declared license CC BY 4.0. No audio ingestion or label audit completed for ARCADE.

## Next action
Start Casablanca ingestion using validation data. Map country configurations to our labels, record duration/channel statistics, and investigate episode/source grouping. W1-03 owns resource/checkpoint provenance; W1-07 owns the final split and overlap audit.

This replaces the earlier requirement to obtain a live NADI endpoint for the main experiment. EEC-8 is ready for review of the revised plan; it does not assert that source independence or pretrained-data independence has already been proved.

## Sources
Checked 2026-10-19:
- [Casablanca release/card](https://huggingface.co/datasets/UBC-NLP/Casablanca) and [metadata](https://huggingface.co/api/datasets/UBC-NLP/Casablanca): access, revision, counts, schema, license.
- [Casablanca paper](https://arxiv.org/html/2410.04527v1): collection and dialect-ID experiment.
- [ADI-17 original resource](https://sls.csail.mit.edu/downloads/adi17/?O=D): YouTube source and labels.
- [Jais-ASR card](https://huggingface.co/IFM/Jais-ASR): Casablanca ASR evaluation.
- [ARCADE release](https://huggingface.co/datasets/riotu-lab/ARCADE-full) and [paper](https://arxiv.org/html/2601.02209v1): alternate-domain candidate.
- NADI phase dates and lack of active phase: user-supplied screenshot and phase text, October 19.

