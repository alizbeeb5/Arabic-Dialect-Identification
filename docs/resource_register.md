# Resource register

## 1. Initial training dataset: ADI-20 Micro

Decision: use this release for the initial training pilots.

- Dataset: UBC-NLP/NADI_2026_ADI20_micro
- URL: https://huggingface.co/datasets/UBC-NLP/NADI_2026_ADI20_micro
- Revision: c8da454c9f474bb075d3fe608940ee29e8ccfa43
- Metadata checked: 2026-10-19
- Access: public and ungated
- Declared repository license: MIT
- Upstream ADI-17/ADI-20 audio-license review: pending
- Training examples: 67,015
- Validation examples: 10,806
- Download size: approximately 21.8 GB
- Published columns: audio, dialect
- Label inventory: all 20 labels in configs/labels.json

### Intended use

Use the training split for learning model parameters.
Use source validation for in-source checks.
Casablanca is the initial target-domain evaluation candidate,
subject to overlap and checkpoint-exposure checks.

### Grouping and overlap limitations

No explicit speaker ID or original-recording ID column is exposed.
A small filename sample has been inspected, as documented below. Broader source-ID recovery and speaker/source independence remain unverified.

### Verification status
Full download, audio loading, and per-class counts remain unverified. Source-ID recovery has been inspected only on a small sample.

### Initial source-ID inspection

Inspected the raw audio.path field in the first training shard
at the pinned revision. This sample contains MSA examples.

Some filenames contain an apparent YouTube recording ID followed
by timestamps. Others contain UUIDs without an obvious source ID.

Conclusion: source-ID recovery appears partially possible.
Filename interpretation must be verified against upstream metadata.
Coverage across other dialects has not yet been measured.

Do not treat a recording ID as a speaker ID.
Keep unrecoverable source_id and speaker_id values null.

Inspection read file metadata and filename columns;
full audio ingestion and overlap checks remain pending.



## Acoustic encoder candidate: AraBEST-RQ-300M-6k

- Model: https://huggingface.co/Elyadata/AraBEST-RQ-300M-6k
- Implementation: https://github.com/elyadata/AraBEST-RQ
- Model revision: dff453e2b5f72f76211a523b1050691967e5a59b
- Access: public and ungated.
- Architecture: approximately 300M parameters; Conformer speech encoder.
- Pretraining: approximately 5,640 hours of crawled Arabic speech.
- Intended use: frozen audio features for the initial classifier baseline.
- Classification head: train separately on our training split.
- Loading: official SpeechBrain implementation; requires model.ckpt
  and normalizer.ckpt.
- License: no explicit license declaration found in model metadata;
  unresolved.
- Status: candidate selected for the pilot; loading, feature extraction,
  dependency compatibility, and hardware requirements remain unverified.
- Pretraining overlap with evaluation recordings: not yet audited.
- 
### AraBEST-RQ download size

- model.ckpt: 1,237,269,115 bytes.
- normalizer.ckpt: 1,917 bytes.
- Combined: approximately 1.24 GB.
- These sizes exclude training-state files such as optimizer.ckpt.
- License remains unresolved: no explicit declaration found.


## Candidate baseline: IFM/Jais-ASR

- Model: https://huggingface.co/IFM/Jais-ASR
- Revision: 1323995fa189c479369a3929be0e95039eba5cba
- Access: public and ungated.
- License: Apache-2.0.
- Coverage: MSA plus 19 Arabic country dialects.
- Intended use: direct dialect prediction and transcript generation.
- Weight download: approximately 11.44 GB, BF16.
- Loading: Transformers with the repository's custom code.
- Official quickstart: transformers==5.16.1, accelerate, soundfile,
  soxr, and hardware-compatible PyTorch.
- Hardware: model card suggests roughly 12 GB GPU memory;
  actual requirements remain unverified.
- Evaluation: use automatic dialect detection; never supply the
  reference dialect as a generation hint.
- Label mapping: Jais codes must be mapped to our canonical labels.
- Full 20-class score extraction: not yet verified.
- Benchmark exposure: Casablanca ASR evaluation is reported;
  training overlap is unknown.
- Status: source inspected; installation and inference not yet tested.



### Jais-ASR to ADI20 label mapping

| Jais output | ADI20 label |
|---|---|
| msa | MSA |
| BH | BAH |
| TN | TUN |
| DZ | ALG |
| EG | EGY |
| IQ | IRA |
| JO | JOR |
| SA | KSA |
| KW | KUW |
| LB | LEB |
| LY | LIB |
| MR | MAU |
| MA | MOR |
| OM | OMA |
| PS | PAL |
| QA | QAT |
| SD | SUD |
| SY | SYR |
| AE | UAE |
| YE | YEM |

Non-Arabic predictions, missing labels, and refusals must be recorded
explicitly rather than silently mapped to an Arabic dialect.



## Initial resource selection

### Training data
- Dataset: UBC-NLP/NADI_2026_ADI20_micro.
- Revision: c8da454c9f474bb075d3fe608940ee29e8ccfa43.
- Use its train split for training and validation split for source
  validation, subject to the grouping/overlap audit.

### Preferred acoustic checkpoint
- Model: Elyadata/AraBEST-RQ-300M-6k.
- Revision: dff453e2b5f72f76211a523b1050691967e5a59b.
- Initial experiment: frozen encoder plus a trained 20-class head.
- Selection remains conditional on license clarification and a
  successful loading test.

### Acoustic fallback
- Model: https://huggingface.co/openai/whisper-small
- Revision: 973afd24965f72e36ca33b3055d56a652f456b4d.
- Access: public and ungated.
- Hugging Face license declaration: Apache-2.0.
- Loading route: Transformers WhisperModel/WhisperProcessor.
- Intended use: frozen encoder plus a trained 20-class head.
- Activate if the preferred checkpoint's terms or compatibility
  prevent its use; record the reason for switching.
- Local loading and pretraining-overlap checks remain pending.
- Coverage: multilingual, including Arabic; a 20-class dialect head
  must be trained separately.
- model.safetensors: 966,995,080 bytes, approximately 0.97 GB;
  processor and configuration files are additional.

### Additional baseline
- IFM/Jais-ASR remains the direct dialect/transcript baseline.

## Released resources versus Thakaa implementation

### Verified from the official NADI 2026 report
- Thakaa ranked first in Task 2 accuracy: 56.15%.
- Its system combined AraBEST-RQ acoustic predictions with text branches.
- Text pipelines used Cohere Arabic and Whisper-large-v3 transcriptions.
- Reported training used approximately 265 hours across eight datasets.

### Available resources for our implementation
- ADI20 Micro training and validation data.
- Released AraBEST-RQ pretrained encoder and implementation.
- Jais-ASR for our direct dialect and transcript baselines.
- Whisper Small as our acoustic fallback candidate.

### Not verified in this audit
- Exact correspondence between our encoder revision and Thakaa's backbone.
- Availability of Thakaa's final trained weights and executable pipeline.
- Its complete training configuration, data manifests, and preprocessing.
- Its exact checkpoint selection and score-calibration procedure.



Source: https://arxiv.org/html/2609.27086v1