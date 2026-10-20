# Acoustic feature interface

Status: implemented; smoke runs verified on CPU, Apple MPS and a Colab Tesla T4.
Code: [src/models/acoustic_encoder.py](../src/models/acoustic_encoder.py),
[src/models/acoustic_smoke.py](../src/models/acoustic_smoke.py),
[tests/test_acoustic_encoder.py](../tests/test_acoustic_encoder.py).

The dialect classifier and the verifier read the same pooled clip
representations. They are computed once by a frozen encoder and cached.

## 1. Checkpoint and loading (verified 2026-10-20)

| Item | Value |
|---|---|
| Model | `Elyadata/AraBEST-RQ-300M-6k` |
| Revision | `dff453e2b5f72f76211a523b1050691967e5a59b` |
| `model.ckpt` SHA256 | `2e9dc6347b93ed7b55c3847c68cfe25947ff4fe21aa7faa9fe5048f15c0ca35a` (1,237,269,115 bytes) |
| `normalizer.ckpt` SHA256 | `30209fa89aad9f7fa007ea2ef17e33ba45c6eacbf7160b581d1946b49dfe6c04` |
| Parameters | 309.24M, all float32; 782 tensors |
| Strict load | 0 missing and 0 unexpected keys |

The architecture is rebuilt from the official recipe,
`elyadata/AraBEST-RQ` `model/hparams/hybrid_300M.yaml` at commit `0ffc8fc`.
`model.ckpt` stores `ModuleList([CNN, EncoderWrapper(TransformerASR)])`.
The quantizer, linear head and optimizer checkpoints are training-only and are not downloaded.

`normalizer.ckpt` has `count = 0` with placeholder global statistics.
This is consistent with `norm_type: sentence`: every clip is normalized by its own Fbank
mean and variance, and no stored statistics are applied. The file is still
loaded and hashed for provenance.

Loading was not blocked, so the Whisper-small fallback was not activated.
The license is still unresolved (see [resource_register.md](resource_register.md)).

## 2. Intended audio processor

Taken from the official recipe:

1. Read the WAV and convert to mono by averaging channels. Resample to 16 kHz if
   needed using `torchaudio.functional.resample`. All ADI20 pilot clips are 16 kHz mono.
2. Compute an 80-bin SpeechBrain `Fbank` with `n_fft=400`, a 25 ms window and a
   10 ms hop (`center=True`). This gives `samples // 160 + 1` frames.
3. Apply sentence-level mean/variance normalization (`InputNormalization(norm_type="sentence")`).
4. Zero-pad to a multiple of 4 frames, as the recipe's `pad_feats` does.
5. Run the CNN front-end (two stride-2 blocks) to get 40 ms frames,
   `ceil((samples // 160 + 1) / 4)` per clip, at 25 Hz.
6. Run the 24-layer Conformer (d_model 848, RelPosMHAXL) with full attention.
   Dynamic-chunk streaming masks are training-time only and are not used.

Cropping: `max_duration_sec=None` by default, so the full clip is encoded. SSL
pretraining used clips of 1–30 s. In the official manifests, every train clip is
30 s or shorter, but 60 dev clips are longer (up to 212.4 s). Those long clips are
untested for GPU memory. Cropping them at 30 s (`--max-duration-sec 30`) changes the
dev features and needs team agreement; the cache key changes automatically.

## 3. Padding invariance

Mask-aware pooling alone does not stop padding from affecting a clip. Stock
SpeechBrain batching leaks padding into a shorter clip's *real* frames in two places:

- The CNN front-end uses reflect padding. In a batch, the clip's last frames
  see batch padding instead of their own reflected edge.
- Each Conformer convolution module masks only its *output*. Its 31-tap depthwise
  convolution still reads padded frames, so about the last 0.6 s of the clip is affected at every layer.

Measured on the real checkpoint, pairing the 3.2 s pilot clip with the 25.7 s clip:

| Path | Max abs difference (batched vs alone) |
|---|---|
| Stock SpeechBrain batching | 0.665 (final-layer cosine 0.986) |
| This implementation | 2.9e-6 (relative 2.2e-7) |

The fix:
- Fbank, normalization and CNN run per clip, so each clip sees exactly the
  padding it would see alone.
- Only the Conformer is batched, with padded frames zeroed before every depthwise
  convolution (`_masked_conv_forward`).
- The encoder's key-padding mask and the pooling mask both use each clip's exact frame count.

For an unpadded clip the output is identical (max diff 0.0) to the stock path,
so this is the official model evaluated as if every clip were alone.
`test_batched_clip_matches_clip_encoded_alone` fails if the fix is removed (max diff 0.78 on the tiny model).

## 4. Feature interface

```python
from acoustic_encoder import AcousticEncoder, FeatureCache, FeatureConfig

encoder = AcousticEncoder.from_pretrained(FeatureConfig(), device="cuda")
result = encoder.extract_manifest(manifest_rows, audio_root, batch_size=8)
cache = FeatureCache(OUTPUT_ROOT, encoder.provenance())
cache.write(result)
features = cache.load(clip_ids)          # same order as clip_ids
```

`ClipFeatures` fields:

| Field | Shape / type | Meaning |
|---|---|---|
| `clip_ids` | list[str], N | Same order as the request; duplicates are rejected |
| `features` | float32 `[N, L, D']` | Masked-mean pooled representation per layer |
| `n_frames` | int64 `[N]` | Real 40 ms encoder frames per clip |
| `layer_names` | list[str], L | Names of the L layer axis |
| `audio_sha256` | list[str], N | Content hash of each source WAV |

Default dimensions: `L = 26`, `D' = 848`. With `pooling="mean_std"`, `D' = 1696`
(masked mean, then population std).

| Layer index | Name | Meaning |
|---|---|---|
| 0 | `frontend` | CNN output after the 1280→848 input projection |
| 1–24 | `conformer_1` … `conformer_24` | Output of each Conformer layer |
| 25 | `final` | Layer 24 after the encoder's final LayerNorm |

`FeatureConfig(layers=(25,))` keeps only the final layer. Both heads accept
`[B, L, D']`. They either select one layer or learn a softmax mixture over L, then:
- the classifier applies a 20-way linear layer;
- the verifier projects the features and scores them against a claimed-dialect embedding.

Both are exercised in `test_both_heads_consume_the_same_pooled_features`.

Leakage boundary: features depend only on audio and `FeatureConfig`. Labels,
paths and grouping metadata never enter them. Heads join labels by `clip_id`.

## 5. Cache layout and keys

```text
OUTPUT_ROOT/features/<encoder>/<revision[:12]>/<feature_config_sha256[:16]>/
    config.json        full provenance: FeatureConfig, checkpoint file hashes, layers
    index.jsonl        clip_id, cache_key, audio_sha256, part, row, n_frames
    part-<hash>.npz    clip_ids, cache_keys, features [n, L, D'], n_frames
```

Per-clip `cache_key` = SHA256 of the canonical JSON of:
- `clip_id` and the WAV's `audio_sha256`;
- `feature_config_sha256`, covering sample rate, resampler, crop, Fbank settings,
  normalization, attention, layers, pooling, precision and code version;
- `checkpoint_revision` and both checkpoint file hashes;
- `FEATURE_CODE_VERSION`.

The cache behaves as follows:
- A changed configuration or checkpoint goes to a new directory.
- A changed WAV under the same clip ID is reported by `missing()`. It is refused
  by `write()` rather than silently overwritten.
- `extract_manifest` checks each WAV against the manifest's `audio_sha256`.
- `load` checks that the index and part files agree.

Bump `FEATURE_CODE_VERSION` whenever a code change alters feature values for the same audio and config.

Size: `26 × 848 × 4 = 88,192` bytes per clip with mean pooling. That is about 6.9 GB
for the 77,821 ADI20 Micro train + validation clips. The small pilot cache is 1.7 MB.

## 6. Numerical tolerance

| Comparison | Tolerance (max abs) | Observed |
|---|---|---|
| Repeated fp32 inference, same device and batches | 1e-4 | 0.0 (CPU) |
| Batched vs alone, fp32 | 1e-4 | 2.9e-6 (CPU) |
| Batched vs alone, fp32, Tesla T4 | 1e-4 | 7.5e-6 |
| Cache write → load (newly written) | exact | 0.0 |
| Cache reuse across batch sizes | 1e-4 | 6.4e-6 (CPU, batch 16 vs 8) |
| CPU vs Apple MPS, fp32 | report only | 3.7e-5 (6 clips) |
| bf16 autocast, Tesla T4: repeat / batched vs alone | 1e-1 | 0.0 / 1.5e-2 |

Pooled values have a magnitude of about 13. Unit tests on the tiny model use 1e-5.
Cross-device and cross-library-version differences are expected at the 1e-5–1e-4
level. Caches should not mix devices without a check.

## 7. Smoke-run record

### Local (CPU), 2026-10-20

- Hardware and software: Apple M3, 16 GB, CPU; Python 3.12.13, torch 2.14.1, speechbrain 1.1.2.
- Data: 20 pilot clips, 140.2 s in total, 3.2–25.7 s each (81–644 encoder frames),
  covering 10 dialects (EGY KSA LEB MAU MSA OMA QAT SYR TUN YEM).
- Model load: 5.5 s. Extraction at batch size 8: 12.2 s, which is 1.64 clips/s or 11.5 audio-s/s.
- Process peak RSS: 4.5 GB. GPU memory: not applicable.
- Checks: all six passed (finite, shape `[20, 26, 848]`, clip IDs preserved,
  repeat within tolerance, padding within tolerance, cache consistent).
- Record: [reports/acoustic_smoke/](../reports/acoustic_smoke/).

### CUDA (Colab Tesla T4), 2026-10-20

Run with [notebooks/W1_07_Acoustic_Smoke.ipynb](../notebooks/W1_07_Acoustic_Smoke.ipynb)
on the same 20 pilot clips as the local run. The checkpoint was downloaded to local
disk and matched both pinned SHA256 hashes. Code: commit `78b4b4f` plus the
checkpoint-hash and cache fixes, which do not change feature values.

| Run | Batch | Precision | Clips/s | Audio-s/s | Peak allocated | Peak reserved | Batched vs alone | Checks |
|---|---|---|---|---|---|---|---|---|
| `20261020T212714361664Z` (reference) | 8 | fp32 | 6.37 | 44.7 | 2.84 GB | 5.69 GB | 7.5e-6 | all passed |
| `20261020T212738124371Z` | 16 | fp32 | 5.11 | 35.8 | 2.84 GB | 10.15 GB | 7.5e-6 | cache check failed* |
| `20261020T212806399920Z` | 8 | bf16 | 4.95 | 34.7 | 2.78 GB | 5.44 GB | 1.5e-2 | all passed |

Repeated inference was identical (difference 0.0) in every run. GB means 2^30 bytes.

\* The batch-16 run reused the 20 clips cached by the batch-8 run, and the old check
required an exact match across batch sizes. The check now allows the tolerance for
clips cached earlier; locally, the same case gives 6.4e-6 and passes.

An earlier fp32 run (`20261020T205105221734Z`, 7.15 clips/s, 2.84 GB) loaded the
checkpoint from Google Drive, whose copy proved unreliable. It is not used as the
reference. Throughput varies by about 10% between runs on shared Colab GPUs.

Conclusions:
- Use **fp32 with batch size 8** on a T4. bf16 is slower on the T4, which lacks fast
  bf16 hardware, saves no memory, and is less precise. Re-evaluate it only on an A100/H100.
- Batch 16 is slower because larger batches carry more padding. Both batch sizes peak
  on the same final batch (the 4 longest clips, up to 25.7 s), which explains the equal
  allocated memory.
- Full extraction estimate on one T4 at 44.7–50.1 audio-s/s, from the official
  manifests: train 163.8 h of audio, about 3.3–3.7 h; dev 25.0 h, about 0.5–0.6 h.
- Peak memory is measured only for clips up to 25.7 s. See the cropping note in section 2.

Rerun on any GPU with the notebook, or directly:

```bash
pip install -r environments/acoustic.txt
python src/data/sample_pilot_clips.py --data-root "$DATA_ROOT"
REV=c8da454c9f474bb075d3fe608940ee29e8ccfa43
python src/models/acoustic_smoke.py \
  --manifest "$DATA_ROOT/manifests/adi20_micro/$REV/pilot-sample/train.jsonl" \
  --audio-root "$DATA_ROOT/audio/adi20_micro/$REV" \
  --output-root "$OUTPUT_ROOT" --cache-root /local/disk/hf-cache --batch-size 8
```

Keep the Hugging Face download cache on local disk. On Google Drive, the 1.24 GB
`model.ckpt` was read back corrupted. The encoder now checks both files against
their pinned SHA256 hashes before loading.

## 8. Pilot sample and open items

- The pilot clips come from `src/data/sample_pilot_clips.py`, because `prepare.py`
  downloads all 21.8 GB even with `--max-clips`. The sample keeps official clip IDs
  but is not an audited release.
- License for AraBEST-RQ: unresolved.
- Pretraining overlap: the model was pretrained on Creative Commons Arabic YouTube
  speech, and ADI20 is also YouTube-derived. Overlap with our training and evaluation
  recordings is not audited.
- Defaults chosen pending team review: store all 26 layers, mean pooling, no crop, fp32.
- The 30 s crop decision for the 60 long dev clips (section 2).
