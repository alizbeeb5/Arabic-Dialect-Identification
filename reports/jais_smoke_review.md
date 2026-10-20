# Task 10 smoke evidence review

Source: the user-uploaded `task10_smoke_bundle.zip`, reviewed 2026-10-21.
The original generated files remain outside ordinary Git history.

The bundle has five training references and five predictions, with no missing
or failed predictions. All five references are MSA. Recomputed accuracy is 2/5
(40%); supported-class macro-F1 is 0.5714285714 over **MSA only**, with 100%
coverage. This is an integration smoke test, not a representative 20-class
evaluation or a comparison with Thakaa. The shared evaluator leaves C_avg
unavailable.

Manifest, label and metric hashes match their declared values. Raw records agree
with standardized predictions. The evaluator source hash matches the supplied
recorded evaluator revision. Both notebook function sources were captured.

The exported run is a later cache-only rerun: five cache hits, zero model batches,
5.5554 seconds wall time and 10.6669 GiB peak allocated GPU memory. Its exported
two-clip cache checks also both reused cached outputs. The missing-audio check
retained one failure alongside one cached output, with no model batch executed.

Earlier chat outputs separately reported an uncached five-clip run taking 11.34
seconds with 10.81 GiB peak allocated GPU memory, and a two-clip inference batch
taking 3.92 seconds with 10.72 GiB. These are user-reported observations; their
original cold-run records are not included in this exported bundle. Do not
attribute those timings to the cache-only exported record or extrapolate them
to the full dataset.

The notebook used the pinned IFM/Jais-ASR checkpoint and FP16 on Tesla T4.
Its direct package versions are recorded in `environments/jais.txt` and the
observed-environment JSON. The source code was consolidated with configurable
paths, stronger cache validation and retained raw evidence on parser failures.
Offline checks and evidence recomputation do not establish that this new
packaged module has run on a GPU; that Colab check remains pending.

## Packaged-source checks

Twelve offline unittest checks passed. The evidence checker recomputed the
shared-evaluator result and verified its declared input/output hashes. The CLI
help command and all six notebook code cells parsed successfully. These checks
ran on CPU without loading Jais or installing Torch.

Evidence bundle SHA256: `464cd908121b354eaf4a8bf12c07f54505e93827b7a0cef8bf9073d8265875b7`.
