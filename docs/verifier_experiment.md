# Controlled classifier-versus-verifier experiment

Task: EEC-19 / W1-15. Status: specification draft; no training or improvement claimed.
Configuration: `configs/verifier_experiment.json` (declarative; no trainer consumes it yet).
Export: [verifier_prediction_export.md](verifier_prediction_export.md).

## Question and scope

Does binary audio-dialect matching improve closed-set ADI-20 identification over
20-way cross-entropy when features, scoring capacity, training examples and
selection budget are identical? This tests a loss formulation, not novelty,
open-set recognition, calibrated verification, or downstream integration.

Both heads use the frozen AraBEST-RQ checkpoint and shared v2 feature interface
in `acoustic_features.md`. This experiment adds no encoder fine-tuning, Jais
fusion, target adaptation, learned layer mixture or hard-negative mining.

## Specification completion versus execution readiness

W1-12 is related work, not a blocker for completing EEC-19. This task delivers
an implementable specification and initial configuration; it does not deliver
a trainer, full feature cache, trained checkpoint or experimental result.
GPU/cache checks below are execution prerequisites, not missing specification
outputs. Different-teammate review remains the issue's completion requirement.
See `../reports/verifier_input_review.md` for the checked input evidence.

## Data and launch gate

Use the same audited, frozen train and development release as the matched C arm;
coordinate those bindings with W1-12 when its implementation is available. Split
original recordings before constructing pairs; keep known speakers separate and
apply the shared duplicate audit. Official split names alone do not certify
recording independence. Record unknown grouping metadata and overlap risks.
Do not silently replace the frozen release with an unaudited pilot.

Before training, resolve and hash: train manifest/metadata, selection-dev
manifest/metadata, label configuration and feature-cache provenance. If an OOD
dev release is used, resolve and hash it too. Prefer the protocol's frozen OOD
selection split. If unavailable, record the agreed in-domain fallback before
viewing candidate results. OOD data is optional for an in-domain experiment;
its absence limits the robustness conclusion and does not prohibit that run.
The JSON's null required release bindings prevent a full experiment launch,
not specification review. `max_duration_sec: null` means full audio, not an
unresolved setting.

The verified reference release is `full-8db5842080c2f643`, with train/dev manifest
hashes in the JSON. The reproducible 500/200 pilots are for implementation checks;
fixing bugs on these pilots must precede the full comparison. Do not apply the
full keep/drop rule to 200 dev clips: one correct prediction changes accuracy by
0.5 percentage points. The final selection release must be bound explicitly;
this document does not certify source or speaker independence.

Use the exact same clip IDs and cache entries in both arms. Fit no transforms
on development. Features use final layer index 25, masked mean pooling, float32,
848 dimensions and no duration crop. The v2 real-checkpoint GPU rerun and handling
of long dev clips must be recorded before full extraction; memory failures must
not cause silent exclusions or arm-specific cropping.

## Identical score function, different training objectives

For a clip feature h in R^848 and candidate dialect c:

    z_c = w_c^T h + b_c

Implement W as a 20-by-848 learned dialect embedding table, with a learned
20-vector bias. A candidate query selects row c; scoring every row gives the
20 logits. This is mathematically the same affine head as a linear classifier.
Each arm has 20*848 + 20 = 16,980 trainable parameters; the encoder is frozen.
Initialize W with Xavier uniform and b with zero. Copy identical initial tensors
into the paired arms for each seed. No sigmoid is applied before the loss.

Classifier C: per-clip cross-entropy = logsumexp(z) - z_y.
Verifier V: form (clip,y,1) and (clip,c,0) for every c != y, then use:

    L_V = 0.5 * softplus(-z_y)
          + (0.5 / 19) * sum_{c != y} softplus(z_c)

Average either loss over original clips. Thus there is one positive and 19
negatives per clip, with total positive/negative weights 0.5/0.5. Enumerate all
negatives; no sampling or mining in the primary experiment. Compute h once and
vectorize the 20 cheap scores. Pair expansion does not create independent audio
observations or 20 optimizer updates. Do not additionally apply pos_weight,
class rebalancing or label smoothing. As a hand-check, when all 20 logits are
zero, C has loss log(20), approximately 2.995732; V has loss log(2), approximately
0.693147. A library BCE mean over the 20 weighted entries would introduce an
extra factor of 1/20 and would not implement the specified reduction.

The same score function isolates the loss comparison. A projected or nonlinear
verifier versus a linear classifier would confound capacity with objective; any
such later experiment must include a matching classifier control and parameter
counts, and be reported separately.

## Fixed optimization and search budget

Use AdamW (betas 0.9/0.999, eps 1e-8), weight decay 0.01 on W only, no bias decay,
clip gradient norm 1.0, batch size 256 original clips, 30 epochs, no scheduler,
no early stopping and no augmentation. Shuffle once per epoch with the same
seed/order in both arms; retain the last partial batch. Seed model initialization
and data shuffling independently from each other using the declared run seed;
consume the same shuffle sequence in paired arms regardless of model RNG calls.
Encoder eval mode, features detached, head float32. Record software/device and
determinism settings. Save epoch checkpoints and validation metrics; reset model
and optimizer state for every trial, including each learning rate.

Search learning rates {1e-4, 3e-4, 1e-3} using tuning seed 11 for each arm:
exactly three trials per arm, with the same number of updates and validation
opportunities. Evaluate after each epoch. Choose checkpoint within each run by
development accuracy, then supported-class macro-F1, then earliest epoch. Choose
learning rate by the selected checkpoints' accuracy, then macro-F1, then lower
learning rate. Freeze each arm's chosen learning rate before the reporting runs.

Run the fixed learning rate afresh with reporting seeds 17, 29 and 43. Apply the
same epoch-checkpoint rule within each reporting run; do not change hyperparameters
after inspecting them. Compute the headline mean, sample standard deviation and
paired gains from these three reporting seeds only. Publish the tuning results
separately and never select the best reporting seed as representative.
Full budget: 6 runs per arm (3 tuning + 3 reporting), 12 total. Also report
measured time, not just updates. Separating seeds avoids including the learning-rate
search seed in the reported variability; the shared dev set still makes these
model-selection results, not an untouched test estimate.

An existing W1-12 classifier is contextual unless it meets these exact controls;
rerun the matched C arm if necessary. Never import a stronger tuning budget
into one side of the controlled comparison.

## Evaluation and robustness

Use the shared evaluator and fixed full denominator, with failures counted as
incorrect. Report development accuracy, supported-class macro-F1, per-label
support/recall/F1, confusion matrix and coverage for every seed and arm. Include
mean and sample standard deviation across seeds and paired V-minus-C differences.

Distribution shift is the intended real-world constraint. When a frozen OOD dev
release exists before tuning, evaluate both arms on identical OOD clips, preserve
all 20 outputs, and report which labels have support. Record before tuning whether
it is the selection set or a separate diagnostic set. A later OOD release is a
follow-up evaluation and must not retroactively change the original keep/drop
rule. Never claim OOD robustness from
different clip IDs alone. Without suitable OOD data, mark source-shift evidence
pending; do not substitute an unsupported robustness claim.

Additionally report clean-development duration slices <=5 s, >5 to <=15 s, and
>15 s, fixed before results. Slice metrics are diagnostics, not additional
hyperparameter selection objectives. Record support; a slice with fewer than
100 clips is descriptive only. Measure all-20-score head inference time and
end-to-end latency on the same device/batch setup, with encoder cost separate.

## Predeclared keep/drop rule

These are project decision thresholds, not established significance criteria.
Keep V as the preferred candidate only if all conditions hold:

1. Mean paired accuracy gain on selection dev is >=0.005 (0.5 percentage points),
   with positive gain for at least two of three seeds.
2. Mean supported-class macro-F1 falls by no more than 0.005.
3. Mean accuracy on every preregistered robustness slice with >=100 clips falls
   by no more than 0.02; include the frozen OOD set if available.
4. Mean coverage is no worse; both arms complete the same planned run budget.

Otherwise retain C as default. Missing runs/invalid releases mean inconclusive,
not a measured loss. If OOD evidence is unavailable, a passing result is a
provisional in-domain keep only. Report variability even when the rule passes;
three seeds are limited evidence. A negative result still answers this task.
No development result establishes a win over a published benchmark score.

## Handoff and implementation acceptance

Deliver this specification, JSON and export specification for teammate review.
Implementation must demonstrate: exact shared feature/ID provenance; 16,980
parameters in each arm; identical paired initial tensors/batch orders; the
explicit weighted loss on a hand-computed example; all 20 candidates at inference;
canonical tie handling; and evaluator validation including failures. The JSON
is an experiment contract, not a runnable training command. Add and document a
trainer CLI in a subsequent implementation task before claiming reproducibility
of training. Keep EEC-19 pending review; do not mark Done merely for drafting.
