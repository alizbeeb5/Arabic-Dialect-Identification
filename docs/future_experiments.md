# Future experiments

## E11 primary comparison

1. Finish the three-seed 4K grid at widths 16, 32, 64, 128, and 256 with β in {0, 0.001, 0.01, 0.1}. Use the same split, cached inputs, and update count for every arm. Select positive β by mean ADI-20 development macro-F1, then source accuracy, smaller width, and smaller β.
2. Carry the selected arm, its matched deterministic and β = 0 controls, and the projection MLP to 1K and 16K. Evaluate all arms on the same ADI-20 dev pilot and all 6,819 Casablanca validation rows. Report paired differences by seed, mean and standard deviation, coverage, and per-dialect results. Record realized KL, trainable parameters, head fit time, and feature extraction time separately.
3. Run EXP-02 and EXP-04 for the supporting layer and corpus figures. Report the grouped probe results and the source-fitted E11 distance AUROC separately. Check whether changes in Casablanca macro-F1 track changes in within-dialect separation.

## Targeted ablations

| If the primary run shows… | Follow-up |
| --- | --- |
| A repeatable KL effect | Repeat the matched 4K comparison on the best single layer selected on source development to test dependence on the input representation |
| Corpus separation concentrated in a few directions | Compare removal of corpus-predictive directions with equally many random directions. Fit on group-disjoint data and record any target-corpus use |
| Strong compression with little active latent capacity | Inspect per-dimension KL and posterior variance before testing annealing or a different prior |
| Persistent Jordanian confusions | Compare matched pairs by recording group, available dialect metadata, and JOR→SYR counts. Include recall and precision on a common evaluation cohort |

## Supporting experiments

| ID | Question |
| --- | --- |
| EXP-01 | How does frozen AraBEST label efficiency change with training size? |
| EXP-03 | Which dialect predictions change under controlled acoustic and temporal perturbations? |
| EXP-05 | Does a supervised contrastive objective change transfer relative to the baseline head? |
| EXP-06 | Can domain-adversarial training reduce corpus sensitivity? Using Casablanca data for fitting makes this an adaptation experiment. |
| EXP-07 | How does temporal pooling affect transfer and cost? |
| Full-cohort fusion | Does the fixed fusion rule improve 4K and 8K audio on the same 6,819 Casablanca rows? Preserve exact audio scores for missing text and report coverage. |

For a final generalization claim, reserve an independent external cohort before architecture selection, and assess recording, speaker, and pretraining overlap to the extent the metadata permits.
