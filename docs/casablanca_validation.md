# Casablanca validation: preparation, inclusion and failure rules
Version: 0.1 | Declared: 2026-10-29, before any Casablanca audio was decoded or any model was scored on it
Task: W2 separate-source development evaluation (related to W2-01; not blocked by it until scoring)

This document fixes the rules first. The audit results live in
[../reports/casablanca_validation_audit.md](../reports/casablanca_validation_audit.md);
scoring results will live in `reports/week2_source_generalization.md`.

## 1. Role of this split
| Item | Rule |
|---|---|
| Primary selection set | ADI-20 source development (unchanged). |
| Casablanca validation | **Frozen diagnostic this week.** It is not used to select checkpoints, seeds, layers, preprocessing, thresholds or ensembles. |
| Casablanca test | Not downloaded, decoded, hashed or predicted by this task. |
| Wording | Because target-validation labels exist and may be used for target-domain development later, any later target-tuned result is "cross-dataset evaluation with target-domain development", never target-blind domain generalization. |

## 2. Source and identity
- Release: [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca), revision `8951b1b88e28c1107142ced57967b8d16350951d`, license CC BY-NC-ND 4.0.
- Only the 15 `*/validation-*.parquet` shards are read. A pattern or file containing `test` is refused.
- Published validation count: 6,819 rows before our audit. A different decoded row count stops preparation for review; it is not silently accepted.
- `clip_id` = SHA-256 of `dataset_id \n revision \n shard path \n row index`, the same construction as the ADI-20 release. `seg_id` is retained as a provenance field only.
- Original parquet files are kept unchanged. Original embedded WAV bytes are written unchanged (no resampling, trimming, channel mixing or normalization at preparation time).

## 3. Label mapping
Labels come from the release's country configuration directory, nothing else.

| Configuration | Label | Published validation clips |
|---|---|---:|
| Algeria | ALG | 844 |
| Egypt | EGY | 846 |
| Jordan | JOR | 848 |
| Mauritania | MAU | 953 |
| Morocco | MOR | 1045 |
| Palestine | PAL | 667 |
| UAE | UAE | 813 |
| Yemen | YEM | 803 |

`evaluation_label_sets.dev` is exactly these eight labels. Models keep all 20 outputs; a prediction of any of the other 12 labels is an error and is never masked or remapped.

## 4. Inclusion and failure rules
1. **Universe.** Every row of the 15 pinned validation shards is an expected evaluation clip.
2. **Decoding.** A row is `preparation_status: ok` when its embedded bytes decode as a nonempty WAV with finite samples and a frame count equal to its header. Otherwise it is `preparation_status: audio_error` with a reason in `exclusions.jsonl`.
3. **Failures stay in the denominator.** `audio_error` rows remain in the frozen manifest. Every model must emit a prediction row with `status: audio_error` for them; they count as incorrect.
4. **No quality-based exclusion.** Short, long, silent, clipped, stereo or duplicate clips are retained and reported. Nothing is removed after looking at model scores.
5. **Exact duplicates.** Clips with identical encoded bytes or identical decoded PCM inside Casablanca validation are reported (with label-conflict flags) and retained.
6. **ADI-20 overlap.** Exact encoded/PCM hash matches against the committed ADI-20 train and dev manifests are reported. The headline keeps all clips; if any match exists, a sensitivity result excluding matched clips is also reported. An absence of exact matches does **not** prove independence, because a resampled or re-encoded copy would not match.
7. **Grouping.** `source_id` and `speaker_id` stay `null` (unknown). The `seg_id` prefix is analysed but not assumed to be an episode, scene or speaker.
8. **Duration disagreement.** The decoded duration is authoritative. Rows whose published `duration` differs from the decoded duration by more than 10 ms are listed, not dropped.

## 5. Audio/preprocessing policy for scoring
Each checkpoint receives Casablanca audio through the same code path and `FeatureConfig` it used on ADI-20 source dev:
- read the extracted original WAV;
- average channels to mono (`channel_mix: mean`);
- resample to the model rate with the model's configured resampler (AraBEST-RQ: 16 kHz, `torchaudio.functional.resample:sinc_interp_hann`);
- encode the full clip (`max_duration_sec: null`), start crop only if a model declares a limit;
- no extra trimming, gain or loudness normalization beyond the model's own.

Feature caches are reused only when the cache key (clip ID, audio SHA-256 and full feature provenance) matches. Casablanca was recorded at 44.1 kHz stereo while ADI-20 is 16 kHz mono, so the downmix/resample step is itself part of the source shift and is reported as such.

## 6. Metadata boundary
Audio is the only predictive input. Human transcripts, `gender`, `seg_id`, country configuration, file names and paths never enter audio-model features or prompts. The frozen manifest does not store transcripts or gender; the audit reports only aggregate counts of them.

## 7. Independence and exposure limits
- Source, episode and speaker independence from ADI-20 and between Casablanca validation and test are **unknown**.
- Checkpoint pretraining exposure to Casablanca is **unknown** for every encoder unless its provenance proves otherwise. Jais-ASR's reported Casablanca ASR evaluation is neither proof of training contamination nor proof of independence.

## 8. Publication
Code, manifests, hashes and results are committed. Casablanca audio, parquet files and transcripts are not committed or redistributed (CC BY-NC-ND 4.0). Cite the dataset.

## 9. Reproduce
```bash
python src/data/prepare_casablanca.py --data-root /path/to/storage
```
Downloads the pinned validation shards only, audits them, extracts original WAVs and writes the frozen manifest and audit. Use `--skip-download` with an existing copy; `--raw-dir` points at a copy outside the default layout.
