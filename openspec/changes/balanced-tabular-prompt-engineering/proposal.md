# Proposal

## Why

The two tabular datasets are imbalanced (30% and 37% positive), which makes accuracy hard to read
and lets a biased model look good or bad for reasons unrelated to skill; on them Jev's default
decision is biased towards the positive class, and nothing has yet been tried to correct it. The
user wants the tabular comparison on balanced data, wants to know whether prompt engineering that
uses training data only can improve Jev, and wants threshold behaviour shown as accuracy against
coverage instead of through a threshold-selection algorithm.

## What Changes

- **BREAKING: the tabular datasets are evaluated on balanced data only.** German Credit and Breast
  Cancer Wisconsin Diagnostic are undersampled to equal classes (300/300 and 212/212 rows, fixed
  seed) and split once into balanced training, validation and test parts, frozen before any new
  model run. This replaces the ten imbalanced repeated splits in the paper. The text datasets are
  unchanged.
- **Prompt engineering for Jev, with the prompt treated as the fitted parameter.** Each round looks
  at Jev's errors on training rows only, records the gap it found and the change it makes, and
  scores the new prompt on training rows. The final prompt is chosen on the validation part and
  run once on the test part. Jev is then reported twice: zero-shot with the original prompt, and
  with the engineered prompt, which is labelled as using training labels. At most 7,000 new Jev
  requests (50,000 of the 60,000 cap are used). Qwen keeps its prompt (the user's decision).
- **Learning curves start at half the training data.** XGBoost and BERT are trained on 50%, 75% and
  100% of the training part, on all four datasets, replacing the 5/10/25/50/100% grid. This needs
  new SageMaker runs of both models (tabular: balanced data; text: the new 75% point).
- **BREAKING: no threshold-selection algorithm.** Cost-based threshold selection (default, analytic
  and validation-selected thresholds, cost matrices, cost sweeps) is removed, and so is the
  validation-selected escalation threshold. In their place the paper shows, for every model and
  dataset, how accuracy and coverage move with the confidence threshold on the test set: items at
  or above the threshold are classified automatically, the rest are left to a person or a stronger
  model. No threshold is selected or recommended. Escalation to Qwen is shown as a curve only.
- The confusion counts (TP, FN, FP, TN) at the default decision stay for the tabular datasets,
  without cost columns.
- The paper, abstract, report and checks are updated; the abstract's sentences about threshold
  selection and the selected escalation point are rewritten from the new results and shown to the
  user before the paper is rebuilt.

## Capabilities

### New Capabilities
- `prompt-engineering`: iterative, training-data-only improvement of Jev's prompt with a recorded error analysis per round, validation-based choice and a single test run.
- `selective-classification`: accuracy and coverage against the confidence threshold on the test set for every model and dataset, with no threshold selected.

### Modified Capabilities
- `experiment-harness`: balanced, singly split tabular datasets; a fifth model entry (Jev with the engineered prompt); new Jev requests allowed within a stage budget.
- `classification-evaluation`: uncertainty for the tabular datasets becomes a bootstrap interval on the balanced test part.
- `supervised-baseline`: learning-curve fractions 50%, 75%, 100%.
- `ensemble-escalation`: the validation-selected threshold is removed; the curve and the ceiling stay.
- `cost-sensitive-thresholding`: retired; all four requirements removed.
- `paper-manuscript`: structure (accuracy-coverage and prompt-engineering sections replace threshold selection); labels used for prompt engineering reported; Jev with the engineered prompt never called zero-shot.

## Impact

- **Jev:** up to 7,000 new requests, about USD 0.15; ledger at most 57,000 of 60,000.
- **AWS:** new XGBoost and BERT executions on all four datasets, estimated USD 2-4 (USD 25.67 of the
  USD 75 limit is spent). No Qwen job: its balanced-test results come from saved predictions.
- **Code:** `data/datasets.py` and `manifest.py` (balanced splits, new freeze entries), new
  `prompt_engineering.py` stage and `experiments/coverage.py`; `experiments/thresholding.py`
  removed; `experiments/escalation.py`, `metrics.py`, `comparison.py`, `baseline.yaml`, packagers,
  `paper/build.py`, `report.py`, `submission.yaml` and the sections.
- **Results:** tabular results of all models are recomputed on the balanced test parts; the
  earlier imbalanced tabular results are copied outside the working copy and no longer reported.
- **Not changed, and stated in the paper:** importance sampling and explanations keep their
  current results, computed with the original prompt on the original rows.
- **Limitation recorded:** aggregate results of the original prompt on all tabular rows, including
  rows that fall in the new test parts, were seen before this change; the engineered prompt itself
  is developed without looking at test rows.
