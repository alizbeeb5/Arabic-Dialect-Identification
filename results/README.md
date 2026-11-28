# Results

These results cover the completed E11 study, baselines, scaling, and diagnostic runs. Macro-F1 averages over the dialects in the reference cohort. Casablanca acoustic models make unrestricted 20-class predictions unless stated otherwise. Casablanca validation has informed earlier experiments and serves as external development.

## E11 variational bottleneck

The source-selected 128-dimensional VIB head with β = 0.01 beat its same-width β = 0 control on full Casablanca at each training size and seed. The largest mean difference was +10.07 macro-F1 points at 1K labels. Read the [E11 protocol, matched comparisons, and corpus diagnostic](e11-vib/README.md).

## Frozen AraBEST scaling

Learned middle-layer mixture, training-only normalization, and 1,000 head updates per seed (490–492). Cohorts are the fixed 200-clip ADI-20 development pilot and full Casablanca validation. One unreadable Casablanca clip remains in the 6,819-row denominator for every run.

| Training clips | ADI-20 accuracy | ADI-20 macro-F1 | Casablanca accuracy | Casablanca macro-F1 | Casablanca JOR recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 86.83% | 0.8696 | 57.99% | 0.6249 | 3.11% |
| 8,000 | 86.67% | 0.8659 | 59.31% | 0.6381 | 1.42% |
| 16,000 | 89.83% | 0.8971 | 60.63% | 0.6544 | 1.49% |

The 8K arm is the same reference in both archives. The 8K→16K difference on full Casablanca was +1.32 percentage points in accuracy and +1.63 points in macro-F1. The aggregate gain coexists with low Jordanian recall. Speaker independence was unverified.

| Export | Run evidence |
| --- | --- |
| 4K vs 8K | [Per seed](scaling-4k-8k/results.csv), [mean and SD](scaling-4k-8k/summary.csv), [paired deltas](scaling-4k-8k/paired_deltas_percentage_points.csv), [protocol](scaling-4k-8k/protocol.json), [confusion counts](scaling-4k-8k/confusion_counts.csv) |
| 8K vs 16K | [Per seed](scaling-8k-16k/results.csv), [mean and SD](scaling-8k-16k/summary.csv), [paired deltas](scaling-8k-16k/paired_deltas_percentage_points.csv), [protocol](scaling-8k-16k/protocol.json), [confusion counts](scaling-8k-16k/confusion_counts.csv) |

## Fixed fusion on balanced Casablanca

This cohort contains 100 clips per dialect and is scored with the unrestricted label space. The text-only model reached 19.88% accuracy and 0.2440 macro-F1. The fusion rule applies text weight 1.3 from the earlier training out-of-fold fit, and uses exact audio scores when usable text is missing.

| Audio training clips | Audio accuracy | Fusion accuracy | Audio macro-F1 | Fusion macro-F1 | Paired macro-F1 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 56.67% | 57.00% | 0.6208 | 0.6238 | +0.30 pp |
| 8,000 | 57.50% | 58.04% | 0.6306 | 0.6356 | +0.50 pp |

Means are across three seeds, each evaluated on the same 800 clips. Paired 4K effects varied in sign, and 8K Jordanian recall stayed at 1.67%. [Per-seed scores](fusion-balanced-800/results.csv), [paired changes](fusion-balanced-800/paired_deltas.csv), [fusion provenance](fusion-balanced-800/fusion_protocol.json), and [confusion counts](fusion-balanced-800/confusion_counts.csv) also cover the ADI-20 dev pilot. The protocol includes the provenance of the full-cohort acoustic checkpoints used in this balanced replay. Full-cohort fusion is a separate experiment.

The earlier [Jais direct tagging method](../docs/jais_pilot.md) on balanced Casablanca reported 391/800 correct, 0.5602 macro-F1, and 99.5% coverage. Direct tag generation and transcript TF-IDF are different classifiers.

## In-source development pilots

All rows use the 200-clip ADI-20 dev pilot. These comparisons used different training settings. Read the within-experiment controls when attributing effects to architecture.

| Model | Accuracy | Macro-F1 | Source |
| --- | ---: | ---: | --- |
| MFCC + SVM | 24.50% | 0.2383 | [Baseline replay](../reports/week1_results.md) |
| Frozen AraBEST, final-layer linear head | 33.50% | 0.3149 | [Baseline replay](../reports/week1_results.md) |
| Jais direct dialect tag | 71.50% | 0.7046 | [Baseline replay](../reports/week1_results.md) |
| Transcript TF-IDF, 500-clip arm | 25.00% | 0.2364 | [Text scaling](../reports/week2_text_scaling.md) |
| Transcript TF-IDF, 2,000-clip arm | 37.50% | 0.3601 | [Text scaling](../reports/week2_text_scaling.md) |
| Layer pilot, standardized final tap | 47.33% | 0.4719 | [Verification](layer-mixture-pilot/verification.json) |
| Layer pilot, selected single tap | 79.00% | 0.7903 | [Verification](layer-mixture-pilot/verification.json) |
| Layer pilot, learned all-layer mixture | 82.83% | 0.8268 | [Verification](layer-mixture-pilot/verification.json) |
| Layer pilot, learned middle mixture | 83.17% | 0.8301 | [Verification](layer-mixture-pilot/verification.json) |
| Layer pilot, learned middle mixture + fixed fusion | 85.17% | 0.8503 | [Verification](layer-mixture-pilot/verification.json) |

The last five rows are three-seed means. The combined pilot used Haidar’s fixed log-space rule with Fatima’s acoustic scores. The pilot verification covers exported predictions and evaluator agreement. It did not rerun extraction from raw audio. On Haidar’s own acoustic rebuild, audio scored 75.50% / 0.7531, probability fusion returned the same score, and fixed log fusion reached 78.00% / 0.7786. [Probability fusion evidence](fusion-source-pilot/fusion_results.json) and [log fusion evidence](fusion-source-pilot/logfusion_results.json) include paired outcomes.

## Other diagnostics

| Diagnostic | Observation | Evidence |
| --- | --- | --- |
| Final-layer residual adapter | 40.50% / 0.3965 for a linear head, 44.00% / 0.4331 for the bottleneck, and 45.50% / 0.4531 with a skip connection. One seed on the 200-clip ADI-20 dev pilot. | [Metrics and fit costs](residual-adapter-pilot/summary.csv) |
| Fatima’s simple frozen head on 80 Casablanca clips | 43.75% accuracy and 0.4991 macro-F1. Convenience sample with its own preprocessing. | [Protocol and metrics](fatima-external-80/metrics.json), [per dialect](fatima-external-80/per_dialect_metrics.csv), [confusion matrix](fatima-external-80/confusion_matrix.csv) |
| Jordanian training augmentation | On full Casablanca, JOR recall averaged 1.49% with 25 original examples, 1.93% with repeats, and 3.22% with 100 distinct examples. At 100 examples, JOR precision fell to 26.33% from 50.24% at 25 examples. Read the [baseline replay](jordanian-data-pilot/baseline_reproduction.csv) when interpreting these small shifts. | [Per seed](jordanian-data-pilot/results.csv), [mean and SD](jordanian-data-pilot/summary.csv), [protocol](jordanian-data-pilot/protocol.json) |
| SHAMS LangMedia, 907 clips | The 4K and 8K acoustic heads averaged 0.3942 and 0.3911 macro-F1 respectively. This cohort contains four shared Levantine labels. Training overlap was unverified. | [Per seed](shams-langmedia/results.csv), [cohort and protocol](shams-langmedia/protocol.json) |
