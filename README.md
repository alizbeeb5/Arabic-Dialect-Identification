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

## Layer 20 ablation and KL controls

E11 found that the KL penalty improved Casablanca macro-F1 by **10.07, 4.60, and 1.17 percentage points** at 1K, 4K, and 16K labels on the layers 6–18 average. The source-selected Layer 20 ablation repeated the 1K comparison with matched training draws, initialization seeds, latent width, and update budget.

| AraBEST input, 1K labels | Casablanca macro-F1, β = 0 | Casablanca macro-F1, β = 0.01 | Paired gain, mean ± SD |
| --- | ---: | ---: | ---: |
| Layers 6–18 average | 0.4338 | 0.5344 | +10.07 ± 2.81 pp |
| Layer 20 | 0.5127 | 0.5744 | +6.17 ± 2.34 pp |

Both rows use the full 6,819-clip Casablanca validation manifest and unrestricted 20-way predictions. Layer 20's gain was positive in all three seeds. See the [verified E11 and Layer 20 results](results/e11-vib/README.md#layer-20-ablation) and [layer and corpus probes](results/exp02-exp04-probes/README.md).

The [KL controls notebook](notebooks/research/ADI20_E11_Layer20_KL_Controls_Kaggle.ipynb) tests whether the improvement comes from penalizing the posterior mean, penalizing the posterior variance, or using input-dependent rather than shared variance. It also compares source-tuned AdamW weight decay. This experiment is ready to run.

For an exact 1K replay, run the original Layer 20 notebook through its training-position definition and export the ordered positions locally:

```python
positions = {
    str(seed): np.frombuffer(zlib.decompress(base64.b85decode(packed)), dtype="<u4").tolist()
    for seed, packed in TRAIN_POSITIONS.items()
}
Path("/kaggle/working/layer20_train_positions.json").write_text(json.dumps(positions))
```

Attach that JSON alongside the feature cache as a Kaggle input. Set `DRAW_NEW_COHORTS = True` to generate fresh balanced draws. Those draws may require additional feature extraction.

### Reuse the AraBEST cache on Kaggle

In your Kaggle notebook, select **Add Input** and attach the [AraBEST pooled feature cache](https://www.kaggle.com/datasets/alizbeeb/adi20-arabest-cache-private). Kaggle mounts the cache files under `/kaggle/input`, where the AraBEST notebooks find them automatically. Check the `to_encode` column before fitting heads. A zero means the mounted cache covers that cohort.

## Repository guide

- [Research notebooks](notebooks/research/) contain E11 and EXP-01 through EXP-07. [Pilot notebooks](notebooks/pilots/) contain earlier scaling, fusion, and diagnostics.
- [Source](src/), [configurations](configs/), and [tests](tests/) contain the reusable baseline, data, and evaluation code.
- [Pilot reports](reports/week1_results.md), [data preparation](docs/data_preparation.md), and the [Casablanca audit](reports/casablanca_validation_audit.md) document the experiment foundation.
