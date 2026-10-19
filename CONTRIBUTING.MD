# Team workflow

## Branches

Create a separate branch for each Linear task.

Examples:
- eec-6-experiment-skeleton
- eec-12-shared-evaluator

Keep main as the reviewed, working version of the project.

## Changes and review

1. Claim the Linear task before starting.
2. Make focused commits on your task branch.
3. Run the relevant checks and record their commands/results.
4. Open a pull request explaining the change and how it was checked.
5. Have a different teammate review it before merging.

## Experiments

Record the configuration, seed, data/split versions, checkpoint revision,
code commit, environment, and output location.

Keep downloaded data, model checkpoints, credentials, and generated runs
out of ordinary Git history.

## Linear handoff

Link the pull request and verification results in the issue.

While awaiting review, keep the issue In Progress with needs-review.
Mark Done after review and acceptance checks are complete.