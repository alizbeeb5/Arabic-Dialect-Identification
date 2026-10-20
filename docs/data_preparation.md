# Task 6 preparation — improved resume and progress

This update preserves the original clip IDs, WAV bytes, manifest rows, and audio
inspection semantics. It adds explicit fast stored-file checks, one-shard local
staging, progress messages, error details, and migration from the original package.
It requires Python 3.10+ and CPU only.

## Continue your interrupted Colab run

1. Stop the old notebook's running cell once; keep the runtime connected.
2. Download this updated ZIP and extract it on your computer.
3. Open the included `notebooks/W1_06_Data_Preparation.ipynb` in Colab.
4. Run its setup, worker-stop, and resume cells in order. Upload the updated ZIP
   when setup prompts for it. Code is extracted into a separate temporary folder.

The stop cell finds only same-user, same-process-namespace `prepare.py` jobs using
this data root. It requests SIGINT, uses SIGTERM if the worker ignores SIGINT, and
waits for exit. It removes only the specified empty stale lock after confirming no
matching workers remain. It never deletes source audio, extracted WAVs, or audits.
V2 handles both interruption signals with status/lock cleanup. An incomplete shard
may be retried; completed shard audits persist.

The resume cell imports the completed audits under the old
`full-06cc7cf3bc843774` report directory. It recomputes the original package's exact
release identity from its known code fingerprint, label file, pinned revision,
shard inventory, selected splits, and limit. If these disagree it stops, rather
than guessing which cached results are compatible. The migration test simulates
14 completed original shards and confirms that their manifest rows are unchanged.
The original report directory stays intact; the new run has a separate report directory.

## What makes this version faster on Drive

- `--verification size` checks that every stored WAV exists and has the expected
  byte length. It avoids reopening every WAV to calculate its content hash again.
  It also avoids reading every newly written WAV back solely for a content hash.
- `--work-root /content/adi20-work` copies and hashes one Parquet shard onto the
  runtime's local disk in one pass. Parquet decoding then reads that local copy.
  The temporary shard is removed after that shard's processing finishes. Only
  approximately one Parquet shard plus 256 MiB of headroom is needed locally.
- Progress identifies the current shard, source copy/hash, checks of saved WAVs,
  and decoded row counts. Updates occur every 100 rows/files or about 10 seconds
  between I/O calls; a blocked individual I/O call can still delay an update.

**Verification tradeoff:** size checks cannot detect corruption that preserves
file size. The audio SHA256/PCM fingerprints still come from fully decoded source
bytes; fast mode does not claim that the stored WAV content was re-hashed.
Reports and metadata explicitly record the chosen mode. Source Parquet SHA256,
label validation, decoding, duration/rate/channel/finite-sample checks, exclusions,
and overlap reporting remain enabled. Fast mode is appropriate for this working
Drive copy; use full verification when validating the final prepared release.
No particular speedup or completion time is guaranteed; Drive metadata and writes
can still be slow. This update has not been timed on your actual Drive.

`--verification sha256` is the default for ordinary team use. Switching from size
to sha256 uses the same v2 audit directory and performs full stored-WAV hashing;
it does not repeat decoding for valid completed shards. A content mismatch stops
for review rather than silently replacing the conflicting file.

## Portable team usage

The team supplies its own writable storage root and runs preparation once before
training workers start. It does not need our Drive account.

```bash
python -m pip install -r requirements-data.txt
python src/data/prepare.py --data-root /shared/arabic-dialect
```

This downloads the pinned Hugging Face release directly there and defaults to
full stored-WAV SHA256 verification. An optional fast local working disk can be
supplied through `--work-root /local-scratch/adi20`.

To reuse an existing notebook/script download offline, add `--skip-download`.
This requires its matching `download_record.json` and all 44 source shards with
recorded sizes. The original record's lack of publisher hashes is retained as
an integrity limitation: locally computed hashes alone do not prove origin.

Other options remain: `--labels PATH`, `--cache-root PATH`, `--splits train dev`,
`--max-clips 5` (first rows per selected split for a pilot), `--batch-size 16`,
and `--workers 2` (download threads only).

## Storage and outputs

Under DATA_ROOT, unchanged original Parquet files live in
`raw/adi20_micro/<revision>/`, and extracted WAVs in
`audio/adi20_micro/<revision>/`. The new code reuses those locations.

Manifests live under `manifests/adi20_micro/<revision>/<release_id>/`, and reports
under `reports/adi20_micro/<revision>/<release_id>/`. The script prints exact paths.
Each selected split gets a JSONL manifest and `.metadata.json` sidecar with its
hash, labels hash, declared audio root, support sets, code provenance, and stored-file
verification mode. Manifest audio paths are relative to the declared audio root.

Reports include `audit.json`, `exclusions.jsonl`, `overlaps.jsonl`, `status.json`,
`shard_state/` receipts, and migration provenance when importing old audits.
Failure status now records the error type, message, and last phase.

V2 release identities use an explicit audit-semantics version plus dataset,
labels, inventory and selection. Logging or I/O changes no longer invalidate
all progress. Developers must bump PIPELINE_VERSION when changing clip identity,
decoding semantics, exclusions, or manifest contents. Actual source code hashes
remain recorded in run reports and sidecars.

Preparation retains upstream splits and reports exact duplicates without removing
them. Near-duplicate analysis, final split freezing, and verified source/speaker
independence remain Task 7 work. Filename-derived recording IDs remain explicitly
unverified; speaker/domain remain unknown. Do not silently remove failed clips
from a fixed evaluation denominator. Exit codes: 0 completed without exclusions,
2 completed with exclusions, 1 preparation error, 130 interrupted.

## Checks

```bash
python -m unittest discover -s tests -p test_prepare.py -v
```

The synthetic offline suite covers the original checks plus legacy migration,
fast/full verification differences, local staging cleanup, actual failure status,
and safely stopping a matching worker. `tests/fixtures/prepare_v1.py` is the frozen
original script used only for compatibility regression checks.

## Current project validation status

The improved implementation passed 19 synthetic checks. The real-data pilot completed with 5 clips and 0 exclusions. The full real-data audit is still running; its final counts, exclusions and overlap findings remain pending review.
