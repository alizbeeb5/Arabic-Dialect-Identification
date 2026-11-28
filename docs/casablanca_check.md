# Casablanca check of the Idea 1 heads

Code: [src/experiments/casablanca_check.py](../src/experiments/casablanca_check.py).
Checks: [tests/test_casablanca_check.py](../tests/test_casablanca_check.py).
Rules: [casablanca_validation.md](casablanca_validation.md).

**Result role:** cross-dataset frozen diagnostic. Heads trained on 500 ADI-20 clips
([layer_mixture.md](layer_mixture.md)) are scored, unchanged, on the frozen Casablanca validation
split (`casablanca-val-v1-90d7d870540d`, 6,819 clips, 8 present labels, TV drama). Casablanca
labels choose nothing: no layer, seed, head or threshold. It is not a benchmark score, and the
Casablanca test split is never read.

## Question

The Idea 1 pilot moved in-source dev accuracy from 47% (last layer) to 79–83% (layer 19 or a layer
mixture). Does that gain survive a change of source: YouTube to TV drama, 16 kHz mono to
44.1/48 kHz stereo, and shorter clips?

## What it does

1. **Features (GPU, once).**
   - Casablanca clips go through the same frozen checkpoint, `FeatureConfig` and code path as the
     ADI-20 pilot (mean downmix, sinc resample to 16 kHz, full clip). They are written into the same
     cache. The script refuses to run if the provenance differs from the pilot run's.
   - Extraction batch size defaults to 1, the W1-12 setting.
   - Cached clips are reused. The encoder loads only when clips are missing, so an interrupted run
     resumes where it stopped.
   - The one row the frozen manifest marks `audio_error` is never decoded. It, and any extraction
     failure, gets an explicit failure row and counts as an error in the 6,819-clip denominator.
2. **Frozen heads (CPU, seconds).**
   - The 15 heads in `run1/arms/` (5 arms × 3 seeds) are loaded unchanged.
   - The z-score statistics are recomputed from the same 500 training clips, which must have the
     pilot's cache-key hashes.
   - Before Casablanca is scored, every head must reproduce its saved ADI-20 dev logits within
     1e-5 and give identical predictions. Otherwise the run stops (`reproduction.json`).
3. **Scoring.**
   - All 20 outputs are kept; predicting one of the 12 absent labels is an error.
   - The shared evaluator scores every head, with macro-F1 over the 8 present labels. The script stops
     if the evaluator's accuracy differs from its own.

## Run it in Colab (T4 GPU runtime)

```python
# Cell 1: mount Drive, unpack the code, check inputs
from google.colab import drive, files
import zipfile
from pathlib import Path
drive.mount('/content/drive')
uploaded = files.upload()  # choose Arabic-Dialect-Identification-idea1.zip
REPO = Path('/content/Arabic-Dialect-Identification')
zipfile.ZipFile('Arabic-Dialect-Identification-idea1.zip').extractall(REPO)
DATA_ROOT = '/content/drive/MyDrive/Arabic-Dialect-Identification-data'
OUTPUT_ROOT = f'{DATA_ROOT}/output'
CAS_AUDIO = f'{DATA_ROOT}/audio/casablanca/8951b1b88e28c1107142ced57967b8d16350951d'
RUN_DIR = f'{OUTPUT_ROOT}/runs/layer_mixture/run1'
print('pilot run found:', Path(RUN_DIR, 'run_record.json').is_file())
print('Casablanca WAVs:', sum(1 for _ in Path(CAS_AUDIO).rglob('*.wav')), '(expect 6819)')
```

```python
# Cell 2: the pinned acoustic environment used for the ADI-20 features
!pip -q install -r {REPO}/environments/acoustic.txt
!python -c "import torch, speechbrain; print(torch.__version__, speechbrain.__version__, torch.cuda.is_available())"
```

```python
# Cell 3: extract Casablanca features and score the frozen heads
CHECK_DIR = f'{OUTPUT_ROOT}/runs/casablanca_check/run1'
!cd {REPO} && python src/experiments/casablanca_check.py --run-dir "{RUN_DIR}" --output-root "{OUTPUT_ROOT}" \
    --audio-root "{CAS_AUDIO}" --cache-root /content/hf-cache --output-dir "{CHECK_DIR}"
```

- **Time:** expect very roughly 45–75 minutes, mostly feature extraction (8 hours of audio, read
  from Drive and resampled). Scoring takes seconds.
- **If Colab disconnects:** rerun the cells with a new output folder (`run2`); clips already cached
  are skipped.
- **If the reproduction check fails after the environment change:** rerun Cell 1 and Cell 3 in a
  fresh runtime without Cell 2. All features are cached by then, so the encoder (and speechbrain)
  is not needed.

Exit codes: `0` completed; `2` completed, but clips beyond the manifest's `audio_error` row failed
(they count as errors); `1` setup or contract error.

## Outputs (in the run folder)

| File | Content |
|---|---|
| `summary.json` | per arm: ADI-20 dev accuracy, Casablanca accuracy and macro-F1 (mean, std, per seed), drop in points, region accuracy, share of predictions to absent labels, recall per dialect, paired comparison with `b0_final` per seed |
| per seed inside `summary.json` | predicted-label counts, absent labels predicted, accuracy by clip duration (0–3, 3–5, 5–10, ≥10 s, unknown), JOR accuracy by native sample rate |
| `reproduction.json` | maximum logit difference of every head against its saved ADI-20 dev logits |
| `arms/<arm>/seed_<s>/` | `predictions.jsonl` (6,819 rows, 20 logits or an explicit failure) and `evaluation/` from the shared evaluator |
| `failures.jsonl` | the manifest's `audio_error` row and any extraction failure |
| `run_record.json` | commit, script hashes, pilot run hashes, manifest hashes, feature provenance and cache-key hash, extraction evidence (frozen encoder, GPU, memory), timings |

## Reading the result

- **Expect a large drop.** The ADI-20 authors lost about 32 F1 points zero-shot on Casablanca. The
  question is whether layer 19 and the mixtures still beat `b0_final` after it (see the paired tests).
- **Absent labels.** A high share of predictions to the 12 absent labels (TUN was the main false
  attractor in NADI 2026 work) means part of the error is class prior, not confusion between the
  8 dialects.
- **Region accuracy** uses a fixed region map (Maghreb, Nile, Levant, Gulf, Iraq, Yemen, MSA). It is
  a diagnostic only.
- **Recording properties correlate with labels** in Casablanca (MAU is the only mono label, JOR the
  only one with 16 kHz audio). Per-dialect differences must not be explained by dialect alone; the
  JOR split by sample rate is the audit's requested check.

## Checks

```bash
python -m unittest discover -s tests -p 'test_casablanca_check.py' -v
```

The tests use a tiny random encoder on synthetic audio and a real `layer_mixture.py` run. They cover:
- an end-to-end run with a manifest `audio_error` row and the shared evaluator;
- a rerun that reuses the cache, never loads the encoder, and gives identical predictions;
- shuffled Casablanca labels leaving predictions unchanged;
- a modified head failing the reproduction check;
- refusal without an audio folder, and refusal of a test split.

Before the first real run, the scoring step was also exercised at full scale: the real 6,819-row
manifest and metadata, random stand-in features, and the 15 heads of a full-size pilot run. All
heads reproduced exactly, and the shared evaluator accepted every row. It took 10 s and 2.5 GB RAM.
