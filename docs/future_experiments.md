# Future experiments

[E11](../results/e11-vib/README.md) is complete. The strongest transfer gain was at 1K labels, with a
+10.07-point Casablanca macro-F1 difference versus the same-width β = 0 control. The within-dialect
distance score showed little change from the input at 1K and 4K, and more separation at 16K.

## Next run

[EXP-02 and EXP-04](../results/exp02-exp04-probes/README.md) are complete. EXP-02 selected
AraBEST Conformer layer 20 on 20-way ADI-20 development. Run the 1K E11 comparison with this
single tap and both KL settings (β = 0 and β = 0.01). Keep the 128-dimensional stochastic heads,
the original training draws, seeds 490–492, 1,000 updates, and full Casablanca validation.
Compare the paired difference with the +10.07-point gain from averaging layers 6–18.

## Targeted ablations

| If the primary run shows… | Follow-up |
| --- | --- |
| A repeatable KL effect | Repeat the 1K, 128-dimensional β = 0.01 versus β = 0 comparison on the best single layer selected on ADI-20 development. Reuse the original train draws, three seeds, update budget, and Casablanca cohort |
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
