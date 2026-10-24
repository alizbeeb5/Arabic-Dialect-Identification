# EEC-19 input readiness review

Reviewed 2026-10-24 against upstream main `174c7c4989e51569190f9ed96f82cd16bc7c056e`
and the local EEC-19 specification branch. Read-only source review; no Drive,
audio, checkpoint or GPU cache was available here. No model training performed.

## Verdict

The frozen pilot manifests are reproducible and usable for an engineering pilot,
with the documented grouping limitations. Full controlled training is not yet
verified ready: v2 real-checkpoint GPU evidence, actual cache coverage/provenance,
long-clip handling and the selection-dev binding remain unresolved. Missing
local evidence is not proof that the team has not generated an artifact elsewhere.

## Independently checked manifests

| Check | Result |
|---|---|
| Full train / dev rows | 67,015 / 10,806 |
| Manifest and label hashes match metadata | Pass, both splits |
| Unique clip IDs within each split | Pass |
| Shared train/dev clip IDs | 0 |
| Shared encoded-audio hash values | 0 |
| Shared decoded-PCM hash values | 0 |
| Shared known source-ID values | 0 |
| All 20 labels represented | Pass, both splits |
| Frozen pilots regenerate byte-for-byte | Pass: 500 train / 200 dev |
| Split version regenerated | `w1-07-pilot-v1-e61cd569ac7b` |

Hashes:

- Full train: `56a75aa929e2a2c4592012954386cb3d91ff7f256a7004ebe7c4a776573ea4ba`
- Full dev: `525ecea4b39c504b28858ddbfaf0a954af703c91ae8edddb252a0a6f2c8d138b`
- Train pilot: `4fb98eabdb15b8bc05fd7bafad0e1c9cd8885a5f50277397159cb2e4bedd65e3`
- Dev pilot: `fb8684c35ef68e25834d5a4c82cddbd31044a899a63b665c41d3f9a179052d02`

Regeneration command (temporary output; compare pilot bytes against repository):

```bash
python src/data/freeze_splits.py \
  --manifest-dir src/data/manifests --dev-split dev \
  --seed 490 --train-per-label 25 --dev-total 200 \
  --output-root /tmp/eec19-split-review
```

These are checks of the manifest contents and recorded audio hashes. Actual WAV
bytes on Drive were not rehashed here. Preparation sidecars record size-only
stored-audio verification; the encoder extraction path separately checks each
WAV SHA256 against the manifest before computing features.

## Leakage limits

Source IDs are present in 5,055 train and 1,794 dev rows: 70,972 of 77,821 rows
(91.2%) have unknown source IDs. No row has a known speaker ID. There are no
cross-split matches among available source IDs, but this cannot establish
recording or speaker independence for the whole release. Near-duplicate detection
was not performed by the freeze script. The report accurately discloses these
limits; do not turn its zero known overlaps into a complete leakage-clearance claim.

The 200-clip dev pilot is balanced and in-source. One additional correct clip
changes its accuracy by 0.5 percentage points; it is primarily an engineering
pilot, not strong evidence of the small gain sought by the full experiment.
Casablanca OOD development is planned, not present in this freeze.

## Encoder and cache evidence

- Code pins AraBEST-RQ revision and checkpoint hashes, verifies source WAV hashes,
  uses masked pooling and padding-invariant batching, and includes preprocessing,
  checkpoint and feature-code identity in cache keys.
- Code is `acoustic-features-v2`. The committed real-checkpoint run record is v1,
  CPU, 20 clips; it reports all checks passed. Documentation also describes v1
  GPU results. Neither establishes real-checkpoint v2 GPU validation.
- No `.npz` feature parts exist in the checkout. Their absence from Git is normal;
  their coverage and values on the team's storage cannot be verified here.
- Full dev includes 60 clips longer than 30 seconds, maximum 212.38 seconds.
  Train maximum is 30 seconds. Previously documented GPU smoke clips reach only
  25.728 seconds. Full-duration inference on long dev clips needs a measured
  memory check; do not drop or crop these silently.

## Checks run here

`python -m unittest discover -s tests -p test_freeze_splits.py -q`:
11 tests passed. Additionally regenerated the real frozen pilots and independently
checked metadata hashes, label inventory, overlap sets and duration counts.

`python -m unittest discover -s tests -p test_acoustic_encoder.py -q`:
could not import because `soundfile` is absent in this runtime. No acoustic test
pass is claimed. A synthetic encoder test pass would still not replace a
real-checkpoint GPU check.

## Concrete next actions

1. W1-12 is unfinished as confirmed by Ali. When its implementation is available,
   inspect it and match its input releases to these hashes. This does not block
   completion of the EEC-19 specification. Training can wait for its reusable
   classifier code or implement the matched C arm in a subsequent task.
2. On the acoustic GPU environment, rerun v2 smoke using the frozen pilot and
   save its `run_record.json`, including all check results, checkpoint and code
   version. Existing CLI:

   ```bash
   python src/models/acoustic_smoke.py \
     --manifest data/manifests/train_pilot.jsonl \
     --audio-root "$AUDIO_ROOT" --output-root "$OUTPUT_ROOT" \
     --cache-root "$CACHE_ROOT" --device cuda --batch-size 8 --limit 20
   ```

3. Inventory actual cache `config.json`, `index.jsonl` and part files. Require
   matching v2 provenance, unique expected IDs, valid cache keys, finite features,
   final-layer availability and no missing expected clips. Read the final layer
   from an all-layer cache if present; do not force unnecessary re-extraction.
4. Test longest dev clips with full duration at batch size 1 first. If a common
   preprocessing change is necessary, record it before results and use it for
   both arms. Bind the chosen selection-dev release before tuning.
5. Use the 500/200 pilots for correctness runs, then perform the matched full
   experiment once the runtime/data gates are resolved. OOD evidence remains
   pending unless a separate audited release is supplied.

No Linear status changes, review requests, pushes or PRs were made by this review.
