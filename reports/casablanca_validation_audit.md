# Casablanca validation audit (W2 separate-source development)

**Split version:** `casablanca-val-v1-90d7d870540d`
**Manifest SHA256:** `5ed7a36491ba6a41782302e1f3d31c3033b4d72ac18f1f7a2d04e85177dd064e` (`data/manifests/casablanca_validation.jsonl`)
**Clip-ID list SHA256:** `11f1d20094ddf0c88338aa45549961be8d2ff53252f89f7285991c0830488790` (`data/manifests/casablanca_validation_freeze.json`)
**Source:** `UBC-NLP/Casablanca` revision `8951b1b88e28c1107142ced57967b8d16350951d`, CC BY-NC-ND 4.0
**Labels SHA256:** `5cf7d68237eab4a39790dda4faf59af6d31bff7c927564e55272e890206902c5`
**Rules (declared before decoding):** [docs/casablanca_validation.md](../docs/casablanca_validation.md), SHA256 `7e4f14d5…f396`
**Preparation code SHA256:** `f72fb296…91a4` (`src/data/prepare_casablanca.py`)

Run on 2026-10-29 in Colab (CPU, 2 vCPU, x86_64), 613.5 s including download, via
[notebooks/W2_Casablanca_Validation_Prep.ipynb](../notebooks/W2_Casablanca_Validation_Prep.ipynb):

```bash
python src/data/prepare_casablanca.py --data-root "$DATA_ROOT"
python src/data/check_casablanca_preprocessing.py --data-root "$DATA_ROOT"
python src/data/analyze_casablanca_grouping.py          # local, manifest only
```

**Status: frozen, ready for review before scoring.** No model has been scored on these clips.

## 1. What was read
- The 15 `*/validation-*.parquet` shards only. Each shard's SHA256 matched the Hub LFS hash (15/15 `hub_hash_verified`).
- **Casablanca test was not downloaded, decoded, hashed or predicted.**
- Original WAV bytes were extracted unchanged to `audio/casablanca/<revision>/dev/<Config>/<clip_id>.wav` under the data root (Drive). Audio and Parquet files are not committed.
- The manifest holds no transcripts or gender. `original_path`, `seg_id` and grouping fields are provenance for auditing only.

## 2. Counts and label mapping
Decoded counts equal the published counts for every configuration (6,819 rows).

| Configuration | Label | Published | Decoded | OK | `audio_error` | Hours | Episode files |
|---|---|---:|---:|---:|---:|---:|---:|
| Algeria | ALG | 844 | 844 | 844 | 0 | 1.00 | 39 |
| Egypt | EGY | 846 | 846 | 846 | 0 | 1.00 | 64 |
| Jordan | JOR | 848 | 848 | 847 | **1** | 1.00 | 55 |
| Mauritania | MAU | 953 | 953 | 953 | 0 | 1.00 | 201 |
| Morocco | MOR | 1045 | 1045 | 1045 | 0 | 1.00 | 89 |
| Palestine | PAL | 667 | 667 | 667 | 0 | 1.00 | 30 |
| UAE | UAE | 813 | 813 | 813 | 0 | 1.00 | 63 |
| Yemen | YEM | 803 | 803 | 803 | 0 | 1.00 | 53 |
| **Total** | **8** | **6819** | **6819** | **6818** | **1** | **8.01** | **594** |

Every configuration holds about one hour (1.000–1.003 h), so the release appears to be capped at one hour per dialect.
`evaluation_label_sets.dev` = ALG, EGY, JOR, MAU, MOR, PAL, UAE, YEM. Models keep all 20 outputs; the other 12 labels count as errors.

## 3. Failure kept in the denominator
| clip_id | Label | seg_id | Reason |
|---|---|---|---|
| `358dae04…a1b` | JOR | `9121_1` | Empty audio or decoded frame count differs from header |

The row stays in the manifest with `preparation_status: audio_error`; its original bytes are extracted so pipelines fail on it explicitly. Every model must emit `status: audio_error` (or its own failure) for it. Headline denominator: **6,819**.

## 4. Audio properties
All 6,818 decodable clips are PCM_16 WAV. No silent clips. Decoded duration agrees with the published `duration` within 10 ms for every clip.

| Label | Sample rates (Hz: clips) | Mono / stereo | Stereo with distinct channels | Median / p95 / max s | < 1 s | > 10 s | > 1% near full scale |
|---|---|---:|---:|---|---:|---:|---:|
| ALG | 44100: 817, 48000: 27 | 0 / 844 | 696 | 3.37 / 10.34 / 40.79 | 20 | 44 | 12 |
| EGY | 44100: 846 | 0 / 846 | 637 | 3.34 / 11.04 / 27.22 | 46 | 52 | 0 |
| JOR | 16000: 450, 44100: 344, 48000: 53 | 0 / 847 | 812 | 3.70 / 8.70 / 17.83 | 11 | 25 | 55 |
| MAU | 48000: 953 | 162 / 791 | 777 | 3.08 / 9.24 / 26.39 | 21 | 40 | 35 |
| MOR | 44100: 1045 | 0 / 1045 | 921 | 3.08 / 7.63 / 17.49 | 75 | 19 | 0 |
| PAL | 44100: 667 | 0 / 667 | 582 | 4.43 / 13.15 / 22.05 | 24 | 73 | 0 |
| UAE | 44100: 619, 48000: 194 | 0 / 813 | 805 | 3.49 / 10.41 / 24.92 | 17 | 48 | 0 |
| YEM | 44100: 488, 48000: 315 | 0 / 803 | 726 | 3.73 / 10.16 / 24.11 | 32 | 43 | 0 |
| **All** | 16000: 450, 44100: 4826, 48000: 1542 | 162 / 6656 | 5956 | 3.45 / 10.03 / 40.79 | 246 | 344 | 102 |

Shortest clip 0.195 s (5 clips under 0.5 s); 2 clips over 30 s. Longest: 40.79 s, 44.1 kHz stereo (ALG), used for the memory benchmark at scoring time.

**Recording properties are label-correlated.** Only MAU has mono clips and it is 100% 48 kHz; JOR is the only label with 16 kHz audio (53%); clipping concentrates in JOR, MAU and ALG. ADI-20-trained checkpoints never saw Casablanca, so they cannot have learned these as shortcuts, but they can still change per-dialect error rates. ADI-20 itself is 16 kHz mono. Two consequences:
1. Per-dialect differences between models must not be explained by dialect alone; report JOR results split by native sample rate as a diagnostic.
2. Any later model **trained or tuned on Casablanca** could exploit sample rate, channel layout or clipping. Such work needs a check for these channel cues.

## 5. Duplicates and ADI-20 overlap
| Check | Result |
|---|---:|
| Exact encoded-audio duplicates inside validation | 0 groups |
| Exact decoded-PCM duplicates inside validation | 0 groups |
| Exact encoded/PCM matches against ADI-20 train (67,015) and dev (10,806) | 0 |
| ADI-20-style YouTube IDs in Casablanca file names | 0 |

**Limitation.** ADI-20 is 16 kHz mono and Casablanca is mostly 44.1/48 kHz stereo, so an exact hash cannot match a resampled, downmixed or re-encoded copy. Zero exact matches therefore does **not** show that the two collections share no recordings. No near-duplicate (fingerprint) comparison was run; it needs ADI-20 audio and was outside this CPU preparation. Shared TV series or speakers between ADI-17/ADI-20 YouTube sources and Casablanca remain **unknown**.

Transcripts (audit only): 13 groups of identical transcripts cover 36 clips; with no audio duplicates these are most likely short common phrases.

## 6. Grouping: episode files and parent segments
`seg_id` prefixes are **not** episodes or speakers. Grouping was recovered from `original_path`, which has the form
`<episode file title>_<start>_<end>_<seg_id>.wav` for all 6,819 clips (`src/data/analyze_casablanca_grouping.py`, outputs `episode_groups.json` / `.jsonl`):

| Finding | Result |
|---|---:|
| `seg_id` equals the file-name suffix | 6819 / 6819 |
| Episode files (filename stem + configuration) | 594 |
| Episode files spanning more than one label | 0 |
| Parent segments (`seg_id` prefix within an episode) | 6,062 |
| Parent segments whose clips all carry one time window | 6,062 / 6,062 |
| Clips sharing a parent segment with another clip | 1,404 |
| Largest single-episode share of a label | 5.7% (UAE) |

Interpretation: the file-name window is the **parent segment's** span; suffixes `_1`, `_2`, … are sub-segments within it (their durations differ, and their offsets are not given). Prefixes never repeat across configurations or episodes. Episode titles show several episodes per series (for example, multiple episodes of one Jordanian series), so **series-level grouping is coarser than these 594 groups and was not derived**.

Limits, all unverified:
- Groups are filename inferences (`inferred_from_original_path_not_verified`), not verified episode metadata.
- Speakers are unknown. Episodes of the same series very likely share actors, so speaker independence does not hold within validation and is unknown elsewhere.
- **Validation/test overlap was not checked**, because no test file was read. Validation and test are likely drawn from the same series. Checking needs test `original_path` metadata (no audio or labels); that is a team decision. It matters only once Casablanca validation is used for target-domain development.
- Sub-segments of one parent segment may overlap in time; this cannot be confirmed without offsets or audio comparison.

Metadata quality (audit only): `gender` values are inconsistent (`M` 4,805, `F` 1,900, `Missing` 83, `None` 26, plus `M,`, `,F`, `M\`, `?`). It is not used anywhere.

## 7. Preprocessing check
`src/data/check_casablanca_preprocessing.py`: the 10 longest and 40 seeded random clips (seed 490; 32 at 44.1 kHz, 13 at 48 kHz, 5 at 16 kHz, all eight labels) through `acoustic_encoder.load_waveform` with the default `FeatureConfig` (mean downmix, `torchaudio` sinc resample to 16 kHz, full clip; config SHA256 `e4b3c287…70bb`).

| Check | Result |
|---|---:|
| Clips passed / failed | 50 / 0 |
| Output length within 1 sample of the expected 16 kHz length | 50 / 50 |
| Finite output | 50 / 50 |
| Longest clip (40.79 s) | 652,590 samples, 0.72 s CPU |
| Wall time / peak RSS | 3.18 s / 349.6 MiB |
| Versions | Python 3.13.18, torch/torchaudio 2.11.0+cpu |

Maximum output peak was 1.043: resampling can overshoot full scale on clipped input. No normalization is added, consistent with ADI-20 handling. Encoder memory and throughput are **not** measured here; they are benchmarked on the scoring hardware before full inference.

## 8. Independence and checkpoint exposure
- Source, episode and speaker independence from ADI-20: **unknown** (exact-hash check only, see §5).
- Validation vs test independence: **unknown** (not checked, see §6).
- Pretraining exposure of AraBEST-RQ, Jais-ASR or other encoders to Casablanca: **unknown**. Jais-ASR's reported Casablanca ASR evaluation is neither proof of training contamination nor proof of independence.
- Results on this split are cross-dataset evaluation. Because labels are available, any later Casablanca-tuned result is "target-domain development", not target-blind generalization.

## 9. Review and reproduction checklist
- [x] Frozen IDs, hashes, label mapping and failure list committed for review before scoring.
- [x] Offline tests: `python -m unittest discover -s tests -p test_prepare_casablanca.py -v` (5 tests).
- [x] Colab run used exactly the local code, rules, labels and ADI-20 manifests (hashes checked on return).
- [ ] A different teammate reruns the notebook (or the commands above) and confirms the same manifest SHA256 and split version.
- [ ] A different teammate reviews this audit.
- [ ] Team decision on checking validation/test episode overlap from test metadata only.

## Files
| File | Content |
|---|---|
| `data/manifests/casablanca_validation.jsonl` | Frozen manifest, 6,819 rows |
| `data/manifests/casablanca_validation.metadata.json` | Sidecar for the shared evaluator |
| `data/manifests/casablanca_validation_freeze.json` | Ordered clip IDs, shard hashes, failure IDs |
| `reports/casablanca_validation_audit/audit.json` | Full machine-readable audit |
| `reports/casablanca_validation_audit/exclusions.jsonl` | The one `audio_error` row |
| `reports/casablanca_validation_audit/duplicates.jsonl`, `adi20_overlap.jsonl`, `duration_mismatches.jsonl` | Empty: no findings |
| `reports/casablanca_validation_audit/seg_id_analysis.json` | `seg_id` structure per configuration |
| `reports/casablanca_validation_audit/episode_groups.json` / `.jsonl` | Inferred episode and parent-segment groups |
| `reports/casablanca_validation_audit/preprocessing_check.json` | 50-clip preprocessing check |

Colab run logs are kept on Drive (`reports/casablanca/` under the data root) and are not committed (`*.log` is ignored). Their results are captured in `audit.json` and `preprocessing_check.json`.
