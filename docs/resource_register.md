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

Next, inspect a small sample to determine whether audio paths retain
original source identifiers. Do not assume source or speaker independence
until the relevant audit is complete.

### Verification status

Repository metadata and published schema checked.
Full download, audio loading, per-class counts, and source-ID recovery
have not yet been verified.