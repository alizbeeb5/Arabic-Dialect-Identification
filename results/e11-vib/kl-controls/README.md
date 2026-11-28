# Layer 20 KL regularization controls

We tested which terms of the Gaussian KL penalty contribute to the 1K transfer result. All arms used frozen AraBEST Conformer Layer 20, the three original class-balanced 1K training draws, 128 latent dimensions, and 1,000 head updates. The stochastic heads sample during training and classify the posterior mean at inference. The shared-variance arm learns one variance per coordinate across all clips. The decay control selects the best of 0.001, 0.01, and 0.1 by mean ADI-20 dev200 macro-F1 and selected 0.1.

| Training arm | ADI-20 dev200 macro-F1 | Casablanca macro-F1 | Paired Casablanca change vs. β = 0 |
| --- | ---: | ---: | ---: |
| β = 0 | 0.8067 | 0.5127 | — |
| AdamW decay 0.001 | 0.8067 | 0.5126 | −0.01 pp |
| AdamW decay 0.1, source-selected | 0.8087 | 0.5137 | +0.10 pp |
| Variance penalty only | 0.8172 | 0.5288 | +1.60 pp |
| Mean penalty only | 0.8087 | 0.5679 | +5.51 pp |
| Full VIB | 0.8170 | 0.5744 | +6.17 pp |
| Full VIB with shared variance | 0.8137 | 0.5748 | +6.20 pp |

Mean-only retains clip-dependent sampling and omits the explicit variance penalty. Full VIB exceeded mean-only by 0.65 ± 1.73 points (mean ± paired seed SD). The difference between full VIB and shared variance was −0.04 ± 0.47 points. These descriptive results cover three historical draws on one external development cohort.

The mean-only head reached 0.5679 macro-F1 with 280.35 nats of average KL to the prior on scorable Casablanca clips. Full VIB reached 0.5744 with 10.62 nats. The KL is an upper bound on input mutual information for the sampled latent under the cohort distribution. The two values alone leave the actual mutual information and nuisance retention undetermined.

Predictions use the unrestricted 20-way label space. The ADI-20 pilot has 200 clips and 20 reference labels. Casablanca macro-F1 uses eight reference dialects across 6,819 validation rows, with one known decode failure counted in evaluation. Both cohorts are development evidence.

[Per-seed scores and KL terms](metrics_per_seed.csv) · [Paired contrasts](paired_contrasts.csv) · [Contrast summary](contrast_summary.csv) · [Per-dialect summary](per_dialect_summary.csv) · [Source selection](source_selection.json) · [Protocol](protocol.json) · [Kaggle notebook](../../../notebooks/research/ADI20_E11_Layer20_KL_Controls_Kaggle.ipynb)

All 42 accuracy and macro-F1 scores were recomputed from the archived predictions. Source archive SHA-256: `20c359b96d2a3e8ffd313bd24cf86f889aaee4a8e08cafe88134f96194be378d`.
