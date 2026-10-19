# Benchmark protocol
Version: 0.1 | Date: 2026-10-19 | Status: draft for team review
Owner: Ali Zbeeb | Linear: [EEC-5 / W1-01](https://linear.app/eece490/issue/EEC-5/w1-01-define-the-benchmark-protocol-and-shared-data-contracts)

## 1. Objective and scope
Build an Arabic spoken dialect identification system with the goal of exceeding Thakaa's reported **56.15% accuracy on NADI 2026 Task 2**, under comparable evaluation conditions. A development score is not evidence of a benchmark win.

The task assigns each audio clip one of the 20 ADI-20 labels. The final benchmark recordings cover six countries and 18 city-level varieties, but this does not define an 18-class output task. Participants were not told which country subset would occur. Preserve the full 20-class inventory; do not mask outputs using final-test label knowledge.

The project studies dialect identification and candidate verifier/Jais contributions. Harness, scoring-API, and downstream training-loop integration are outside scope. This protocol specifies evaluation rules, not a training implementation.

## 2. Verified benchmark facts and open access questions
| Item | Evidence / decision |
|---|---|
| Target | NADI 2026 Task 2, Spoken Dialect Identification [S1, S2] |
| Comparison target | Thakaa: 56.15% accuracy [S1, S2, Table D.2] |
| Official metrics | Accuracy and average cost [S2] |
| Thakaa cost | Table D.2 reports 11.31 as C_avg multiplied by 100, equivalent to 0.1131 on the unscaled convention [S2] |
| Output inventory/order | Organizer baseline notebook, Submission Format [S3]; inventory corroborated by dataset card [S4] |
| Official-test access | Not verified for this team; tracked by W1-04 / EEC-8 |
| Eligibility | Permitted data/models and current evaluation route require W1-04 confirmation |

A comparable claim requires the same benchmark test set/version, compatible scoring, the full required output inventory, and compatible data/model eligibility and adaptation conditions. If an organizer-sanctioned evaluation is unavailable, report local results as local evaluation and keep the published Thakaa result contextual. A local score exceeding 56.15% on another dataset is not a win over Thakaa.

## 3. Labels
The authoritative project mapping is [../configs/labels.json](../configs/labels.json).
It contains 20 labels, zero-based IDs, both mapping directions, display names, and provenance.

Order:
MSA, BAH, TUN, ALG, EGY, IRA, JOR, KSA, KUW, LEB, LIB, MAU, MOR, OMA, PAL, QAT, SUD, SYR, UAE, YEM.

MSA is Modern Standard Arabic; the other 19 codes represent country-level dialect classes. Reorder each checkpoint's native outputs to this mapping before score comparison, combination, or export. Never assume pretrained model IDs already match.

## 4. Data boundaries and source shift
- **Training:** may update parameters, fit learned preprocessing, and supply training augmentation.
- **Development:** may select checkpoints, hyperparameters, preprocessing choices, calibration, and ensemble weights. It is selection data, not an unbiased final evaluation.
- **Final test:** only evaluates the frozen system. Do not use labels, leaderboard feedback, or error inspection to select the system reported as untouched-test performance.

Distinguish in-domain development from out-of-domain development. Different recording IDs alone do not prove domain shift; define the source/domain difference in the split release.

Rules:
1. Group clips from one original recording/episode into one split. Never randomly scatter segments from that recording across splits.
2. Keep known speakers separate across training and evaluation wherever feasible. Record missing speaker identities and any unavoidable overlap; do not claim verified speaker independence when it is unknown.
3. Check exact and near-duplicate audio across splits. Record the audit method, findings, and exclusions.
4. Fit learned normalization and other data-dependent transforms on training data only.
5. Use identical evaluation clips, label mapping, and selection policy across compared models. For controlled component comparisons, also hold training data, split, and relevant compute/training choices fixed or explicitly report differences.
6. Filenames, source IDs, speaker IDs, and dataset metadata are not predictive input features. Labels are targets. Any transcript branch must use transcripts derived from permitted audio, without label-bearing metadata.
7. Release versioned manifests and evaluation ID order before experiments. Record their checksums and evaluation label sets. This task defines the rules; W1-07 / EEC-11 constructs and freezes actual splits.
8. Do not inspect hidden final-test labels to design splits or choose a target-country subset. Record known pretrained-data overlap and unresolved contamination risks.

## 5. Metrics and failure handling
### Primary accuracy
Accuracy = number of correct predictions / number of expected labeled evaluation clips.
Store the fraction in [0,1] and display percentage = fraction * 100. Use the complete fixed denominator, including failed/missing predictions.

### Per-dialect results
Report support, precision, recall, and F1 for each label, plus a confusion matrix.
For class c:
- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2*TP / (2*TP + FP + FN).
Use zero for undefined precision/F1 denominators, with support and a zero-support flag visible; do not interpret an unsupported class as measured performance.

### Macro-F1 convention
For each labeled evaluation split, preregister the set E of labels with positive reference support before comparing model results. All models use that same set. Macro-F1 is the arithmetic mean of F1 over E. Report E and its size with the result.

Keep all 20 labels in the output vocabulary and per-class table. Labels with no reference support are marked unsupported and excluded from the primary supported-class macro-F1. If a fixed-20 macro-F1 is additionally reported, name it separately and document zero_division=0. Never substitute one convention for the other without labeling it.

A prediction outside E but inside the valid 20-label inventory is still an error for the true class. It is not discarded or remapped. Do not derive the final-test subset from hidden labels for model use; the evaluator may determine support after the model is frozen.

### Operational failures
Each expected clip must have a prediction record. Explicit failures and missing rows count as incorrect for accuracy and as false negatives for their true classes. They are not assigned a real dialect for metric computation. Show a separate failure column in the confusion matrix.

Report expected clips, successful clips, failures by status, missing rows, correct clips, and coverage (successful/expected). Missing rows are evaluator-detected contract violations and are recorded separately from declared model errors. Duplicate IDs, unexpected IDs, invalid labels, and malformed records fail validation; do not silently repair them and publish a score.

### Average cost (C_avg)
C_avg is a separate score-based benchmark metric; do not approximate it from hard labels or replace unavailable scores with fabricated one-hot vectors.
Requirements:
- Finite 20-class scores with explicit label order and score type.
- Any score conversion/calibration chosen using development data only.
- Exact scorer version, dependencies, input transformation, threshold/prior conventions, and scaling recorded.
- Compatibility checked against official reference examples or organizer-confirmed scoring before a benchmark-comparable claim.
- If required scores are unavailable, incomplete, or invalid, report C_avg as unavailable with the reason. Do not compute it only on successful clips and present it as a full-set result.

The inspected organizer notebook [S3] contains two compute_ave_cost definitions with different class-count assumptions (20 versus an 8-class example, including a hard-coded np.eye(8)). This is a reason to verify the actual platform scorer, not to treat the notebook as a frozen, authoritative scoring implementation. Resolving scoring compatibility remains open. Accuracy evaluation can proceed independently.

## 6. Checkpoint and model selection
Proposed team policy:
1. Select the checkpoint with highest unrounded accuracy on the fixed out-of-domain development set.
2. Break exact accuracy ties by unrounded supported-class macro-F1.
3. Break remaining ties by earliest saved training step for checkpoints within a run.
4. For cross-run/model ties, use lower measured inference cost under the same measurement setup; predeclare a deterministic run-ID tie-break if costs also tie.

If suitable out-of-domain development data is unavailable, explicitly approve and record a fallback selection split before comparing candidate results. Do not change the selection set because a model performs poorly on it.

Record the planned search budget and seeds. Compare the same seed set where stochastic training is involved and report variability; do not present a cherry-picked seed as representative. Select preprocessing, calibration, and ensemble settings using development data.

Before final evaluation, freeze checkpoint hashes, code commit, preprocessing, label mapping, inference config, and ensemble/calibration settings. Record all final submissions/evaluations. Later revisions guided by test results must be labeled as test-informed experiments rather than untouched-test performance.

## 7. The 50-target-clip adaptation question
The course announcement requires a realistic constraint approved during project review and lists distribution shift and limited labeled data as possible constraints [S5, p.4]. It does not mandate exactly 50 clips.

Draft scope decision: distribution shift is our primary proposed constraint. The original 50-target-clip adaptation experiment is optional pending confirmation of any project-specific instructor/team commitment. This decision is not evidence that approval has already been obtained.

If retained:
- Treat the 50 labeled target-domain clips as adaptation training data, not development or final-test data.
- Specify whether 50 means total or per class before sampling; do not leave the budget ambiguous.
- Freeze the sampling seed, class allocation, provenance, and source/speaker exclusions.
- Use the same adaptation examples for every controlled comparison.
- Keep adaptation, development, and final-test recordings separate.
- Report adapted and unadapted systems separately and verify eligibility before comparing an adapted system with Thakaa.
- The target adaptation budget does not cap source-domain training data.

## 8. Reproducibility record
Every experiment must record: run ID, code commit, data and split versions/checksums, label-config checksum, model/checkpoint revision, training and inference configs, seeds, exact commands, selection split/rule, metrics and coverage, hardware/software environment, measured runtime/memory and resource costs, and external data/model provenance.

Link predictions and evaluator output to this record. Hashes and immutable revisions must identify actual artifacts, not moving aliases such as "latest".

## 9. Review and handoff
This draft is not yet a reviewed completion of EEC-5.

- [ ] All three members review this same protocol version.
- [ ] Confirm source-shift constraint approval and the 50-clip scope decision.
- [ ] Identify/freeze the out-of-domain development set or explicitly agree on a fallback.
- [ ] Resolve official scoring compatibility, including C_avg conventions.
- [ ] W1-04 records official-test access and data/model eligibility.
- [ ] A different teammate reviews output schemas and label mapping.
- [ ] Link repository files/PR in Linear and record reviewer feedback.
- [ ] Release downstream work only when its other blockers are also complete.

Official-access uncertainty need not block public-data pilots. Keep benchmark claims conditional until resolved. No team review, messages, or Linear state changes have been performed by creating these files.

## Sources
- [S1] [NADI 2026 official site](https://nadi.dlnlp.ai/2026/), Task 2 leaderboard.
- [S2] [NADI 2026 findings, v1](https://arxiv.org/html/2609.27086v1), sections 3.2 and 5.2, Table D.2.
- [S3] [Organizer Spoken Dialect ID evaluation notebook](https://colab.research.google.com/drive/1oxF7CyEGJwhDMWmAoeg-OYSDh6kLsUfW), last updated 2026-06-16; Submission Format and cost examples inspected 2026-10-19.
- [S4] [Organizer ADI-20 Micro dataset](https://huggingface.co/datasets/UBC-NLP/NADI_2026_ADI20_micro).
- [S5] EECE490_Fall_2026_27_Project_Announcement.pdf, supplied course announcement, especially p.4 (real-world constraint) and p.5 (deliverables).

