# Task 13: verified Jais pilot results

Reviewed 2026-10-24. EEC-17 / W1-13. Status: GPU pilot complete and evidence
recomputed; independent teammate review pending. No Jais fine-tuning performed.

## Evidence and reproduction

Source: user-supplied `task13_evidence.zip`, 20 files, 1,447,642 uncompressed bytes.
Archive SHA256: `d4c0808ed6ca286d496f224f49386ea55a995bce6c41ebdd11d90e272314b1bd`.
The original bundle and raw transcripts stay outside Git. Team storage:

`/content/drive/MyDrive/Arabic-Dialect-Identification-data/runs/jais_task13/20261024T185108716428Z`

Train run/export suffix: `20261024T185714935356Z`.
Dev run/export suffix: `20261024T191346757507Z`.
Code commit recorded in both runs: `26b36593c7b4857b34a0b5f864c409f6d546be91`,
clean checkout. Model: `IFM/Jais-ASR`, revision
`1323995fa189c479369a3929be0e95039eba5cba`.

Reproduce the review without audio/model loading:

```bash
python tests/check_jais_pilot_evidence.py /path/to/task13_evidence.zip
```

The checker reproduces all five export files for each split byte-for-byte;
checks adapter/exporter/evaluator identity through the recorded source hashes;
verifies manifest, metadata, prediction and artifact hashes; checks all expected
IDs and raw/standard prediction agreement; recomputes all evaluator fields and
checks the confusion CSVs. These checks reproduce records and scores; they do
not independently assess transcription accuracy against audio or attest GPU
measurements. No missing clip is silently dropped.

Frozen split: `w1-07-pilot-v1-e61cd569ac7b`, 25 train and 10 dev clips per label.

| Artifact | SHA256 |
|---|---|
| Train pilot manifest | `4fb98eabdb15b8bc05fd7bafad0e1c9cd8885a5f50277397159cb2e4bedd65e3` |
| Dev pilot manifest | `fb8684c35ef68e25834d5a4c82cddbd31044a899a63b665c41d3f9a179052d02` |
| Train predictions | `e38375508a656c50c965dffa7fdeb07d9ebb29e661e355e8924f030b10996302` |
| Dev predictions | `daaa9ec6513ac62bfd401517e16f29f5d575c752e48b2fd4040c8959c254f422` |

## Direct Jais dialect prediction

All 20 dialects have positive support, so supported-class macro-F1 averages all
20 here. Training-subset results are pretrained-model inference results, not
performance after fitting Jais on these 500 examples.

| Metric | Train subset | Development pilot |
|---|---:|---:|
| Expected clips | 500 | 200 |
| Correct dialect predictions | 314 | 143 |
| Accuracy, full denominator | 62.80% | 71.50% |
| Macro-F1 | 0.637798121685272 | 0.7045528323025294 |
| Successful dialect predictions | 486 | 200 |
| Dialect failures | 14 | 0 |
| Missing rows | 0 | 0 |
| Coverage | 97.2% | 100% |

C_avg remains unavailable: the adapter exports hard tags without verified
20-class scores, and the official scorer is not implemented/verified.

## Transcript and failure analysis

Training: 492 usable transcripts and 8 refusals. Development: 196 clean
transcripts and 4 explicitly truncated transcripts. All rows remain exported.
No truncation flag is unknown. The exports contain parsed text only; generated
dialect tags, targets and source metadata are separate. Transcript accuracy
against the speech was not manually evaluated by this bundle review.

The six training mapping failures are actual unsupported native-language tags:
`en` (3), `fr` (2), and `es` (1). They are retained as unrecognized-dialect
failures, never remapped to MSA or another country. Their transcripts remain
available to the text branch; this does not prove those transcriptions are
correct. They occur on KSA (2), ALG (2), SUD (1) and SYR (1) references.

The eight refusals occur on OMA (3), SYR, MAU, MOR, BAH and KUW (one each).
No audio read/decode or generation/parse errors appear in these runs.

All four truncated dev transcripts are MSA references, durations 60.04, 61.92,
62.07 and 142.05 seconds. Each reached 128 generated tokens. Three still have
correct MSA dialect predictions; one predicts IRA. They contribute normally to
the 200-clip direct-dialect evaluation. Dropping them from the text evaluation
would remove 40% of its MSA references and change the comparison.

Largest directed dev confusions: TUN -> ALG (6/10 TUN clips), BAH -> MSA (6/10
BAH clips), and KUW -> QAT (5/10 KUW clips). Dev recall is BAH 10%, TUN 30%,
KUW/UAE 40% and MSA/QAT 50%. JOR, ALG and LIB have 10/10 recall. Each class
has only ten examples; these are pilot diagnostics, not population estimates.
Full per-dialect metrics and confusion matrices are in the evidence exports.

## Measured resources

Both run records report a Tesla T4, FP16, greedy decoding, left padding,
batch size 2 and an explicit 128-token generation cap. The checkpoint's stored
generation defaults list 256 tokens, but the adapter's explicit call overrides
that value with 128; both identities are preserved. Environment: Python 3.13.16,
Torch 2.11.0+cu130, Transformers 5.16.1, Accelerate 1.15.0, NumPy 2.1.3,
SoundFile 0.14.0 and soxr 1.1.0. This is a recorded environment, not a complete
transitive lock.

| Measurement | Train subset | Development pilot |
|---|---:|---:|
| Adapter wall seconds | 980.731519389 | 486.472705717 |
| Wall time, approximately | 16 min 21 s | 8 min 6 s |
| Cache hits | 20 | 0 |
| New model outputs | 480 | 200 |
| Model batches | 240 | 100 |
| Peak allocated GPU memory, GiB | 11.0573 | 11.8549 |

The 20 train cache hits are the prior smoke subset. Dev is entirely fresh;
its cold end-to-end throughput is 0.4111 clips/s. Wall time includes loading,
audio/cache reads, preprocessing, generation and cache writes; final export is
excluded. No pure generation GPU-time measurement exists. Do not call the train
run wholly cold or its wall seconds GPU-only compute time.

## Task 14 handoff and remaining review

Use the two saved `transcripts.jsonl` files as text records, joined by clip ID to
separate `targets.jsonl`. Available training text: 492 rows; the 8 refused rows
must have an explicit training policy. Keep all 200 dev IDs when scoring the
text classifier. Before fitting, define whether the four partial dev transcripts
are used as available partial text or receive an explicit fallback; never
silently filter them. Apply any partial-text rule consistently across splits.
Do not tune a different Jais generation limit on these errors and present the
new result as the unchanged baseline. Preserve this release before follow-ups.

This is an in-source ADI-20 pilot. The recording/speaker independence limits in
`split_audit.md` remain. No official-test, OOD, full-dataset or Thakaa comparison
claim follows from 71.5% here. The acoustic model is unfinished; its future
comparison must use exactly the same frozen 200 development IDs.

The implementation, GPU inference, exports, failure analysis and resource record
are ready for review. A different teammate still needs to review before EEC-17
is marked Done. GPU success does not retroactively change the explicit failures
or fill the missing benchmark score vectors.
