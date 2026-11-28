# Reproducing the experiments

## E11 on Kaggle or Colab

| Runtime | Notebook | Persistent inputs and outputs |
| --- | --- | --- |
| Kaggle | [AraBEST experiments for Kaggle](../notebooks/research/ADI20_AraBEST_All_Experiments_Kaggle.ipynb) | Add existing cache parts as input if available. Download the milestone ZIPs from the run output to resume in another session. |
| Colab | [AraBEST experiments for Colab](../notebooks/research/ADI20_AraBEST_All_Experiments_Colab.ipynb) | Mount Drive. The settings cell gives the cache path, optional input folder, and run output folder under the project directory. |

1. Enable a GPU and network access. Kaggle can use separate GPUs for head fitting and encoding. Either notebook can run on one GPU. Use your own Hugging Face account and runtime secret `HF_TOKEN` if dataset access requires one.
2. Keep `RUN = (11,)` for the primary comparison. Set `PREFLIGHT_ONLY = True` and run the notebook to check the pinned manifests, planned cohorts, and cache coverage. The notebook can read the checksum-verified manifests from its pinned GitHub checkout. It also accepts the historical full-manifest bundle.
3. Restore `PREFLIGHT_ONLY = False`. With `E11_ENCODE_MISSING = True`, extraction fills missing feature rows from the pinned dataset revisions. A matching cache speeds up reruns, and the cache records provenance and per-clip audio identity.
4. Save each completed E11 stage archive and the run record. The exports include the 4K sweep, source-dev selection, checkpoint provenance, seed-level metrics, paired comparisons, corpus separation, and per-dialect scores. Reattach compatible fitted-head archives and cache parts when resuming.
5. Use `RUN = (2, 4, 11)` for the paper experiments, or select additional experiment IDs from the launch guide. Some supporting runs consume more storage or use previously fitted heads. The notebook settings document their prerequisites.

## Cohort assignments

The public notebooks generate balanced, nested training anchors and a balanced Casablanca diagnostic cohort from the pinned manifests. The original experiments used frozen historical membership. The public default therefore reproduces the experimental protocol with fresh training membership and fresh balanced Casablanca membership. It keeps the fixed ADI-20 development pilot and the full 6,819-row Casablanca E11 cohort.

For an exact replay, set `COHORT_ASSIGNMENTS_FILE` to a local JSON containing:

- `train` with keys `"2000"`, `"4000"`, `"8000"`, and `"16000"`. The value at each key is the list of zero-based row positions in the pinned ADI-20 training manifest for that size.
- `casablanca_balanced` with the 800 zero-based row positions in the pinned Casablanca validation manifest.

The notebook checks membership counts, class balance, nesting, and known split overlap. Keep the assignments local. For the EXP-03 perturbation study, supply the matching local assignments when reusing historical heads. Without those heads, the notebook refits the 8K head on the selected cohort.

## Source and evaluation checks

The notebooks pin the historical source revision `b64437a37e7eda2f1721a56860278640bf085bdd` to preserve the encoder and classifier implementation used in completed scaling runs. The [dataset protocol](protocol.md) documents the earlier pilot design. The [Casablanca audit](../reports/casablanca_validation_audit.md) records the frozen external-development manifest and its known limitations.

Run the shared evaluator tests and regenerate the original pilot comparison from repository evidence:

```bash
python -m unittest discover -s tests -p 'test_evaluate.py' -v
python src/analysis/week1_results.py --output-dir /tmp/adi20-week1-replay
```

The [acoustic environment](../environments/acoustic.txt), [text environment](../environments/text-pilot.txt), and [Jais environment](../environments/jais.txt) document the original workflows. Older [pilot notebooks](../notebooks/pilots/) reproduce earlier run designs and can require the original input manifests, transcripts, caches, or checkpoints named in their setup cells. The current E11 workflow and published aggregate evidence are accessible independently of those bundles.

Build the working paper with `latexmk -pdf -cd paper/main.tex`.
