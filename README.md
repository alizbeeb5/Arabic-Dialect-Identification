# Cross-corpus Arabic dialect identification

EECE 490 research project at the American University of Beirut

Ali Zbeeb · Fatima Karnib · Haidar Madi

**Research question.** Does a variational information bottleneck improve cross-corpus dialect identification from frozen AraBEST speech features when labeled data is limited? E11 compares KL-regularized heads with controls for dimensional compression and stochastic training.

We fit heads on ADI-20 Micro and evaluate on ADI-20 development and Casablanca external development. Casablanca has already informed earlier project comparisons. All Casablanca scores here are development evidence.

## Results so far

| Experiment | Evaluation | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Frozen AraBEST, learned middle-layer mixture, 16K labels | ADI-20 dev pilot, 200 clips | 89.83% | 0.8971 |
| Same 16K protocol | Full Casablanca validation, 6,819 clips | 60.63% | 0.6544 |
| Frozen AraBEST, learned middle-layer mixture, 8K labels | Full Casablanca validation, 6,819 clips | 59.31% | 0.6381 |
| 8K audio + Jais text fusion | Balanced Casablanca validation, 800 clips | 58.04% | 0.6356 |

Scores in this table are means over seeds 490–492. On the **same balanced 800 clips**, 8K audio alone reached 57.50% accuracy and 0.6306 macro-F1. With 16K labels, Jordanian recall on full Casablanca was still 1.49%. See [results and cohort details](results/README.md) for the baselines, uncertainty across seeds, fusion, and additional diagnostics.

## E11 and supporting experiments

| Resource | Contents |
| --- | --- |
| [Architecture](docs/architectures.md) | Frozen representation, matched VIB controls, and earlier acoustic and text models |
| Paper | [Working manuscript (PDF)](paper/ADI20_Paper_Draft.pdf) and [LaTeX source](paper/main.tex) |
| [Reproduce](docs/reproduction.md) | Run on Kaggle or Colab with generated cohorts or local assignments for exact replay |
| [Future experiments](docs/future_experiments.md) | E11 priorities, ablations, and conditions for stronger transfer claims |
| [Evaluations](results/README.md) | Saved metrics, paired deltas, provenance, and error analysis |

**E11 is complete.** The KL penalty improved full Casablanca macro-F1 versus the same-width stochastic control by **10.07, 4.60, and 1.17 percentage points** at 1K, 4K, and 16K labels. See the [matched results and corpus diagnostic](results/e11-vib/README.md). The completed [EXP-02 layer probes and EXP-04 within-dialect corpus probes](results/exp02-exp04-probes/README.md) select Conformer layer 20 for the next ablation. EXP-01 and EXP-03/05/06/07 are available in the research notebooks.

## Repository guide

- [Research notebooks](notebooks/research/) contain E11 and EXP-01 through EXP-07. [Pilot notebooks](notebooks/pilots/) contain earlier scaling, fusion, and diagnostics.
- [Source](src/), [configurations](configs/), and [tests](tests/) contain the reusable baseline, data, and evaluation code.
- [Pilot reports](reports/week1_results.md), [data preparation](docs/data_preparation.md), and the [Casablanca audit](reports/casablanca_validation_audit.md) document the experiment foundation.
