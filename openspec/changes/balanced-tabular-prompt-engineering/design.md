# Design

## Context

See proposal.md for the motivation. Facts that shape the approach, as of 2026-10-02:

- The tabular datasets use ten stratified 60/20/20 repeats over all rows (`protocol: repeated`).
  Jev and Qwen have one saved prediction for every row with the original prompt, so their results
  on any subset of rows cost nothing. XGBoost and BERT are fitted per repeat on SageMaker.
- German Credit has 700 good and 300 bad rows; Breast Cancer Wisconsin Diagnostic has 357 benign
  and 212 malignant.
- From the saved predictions: Jev's balanced accuracy at the argmax is 0.565 (German Credit) and
  0.538 (Breast Cancer Wisconsin); the best any threshold could give is 0.568 and 0.840. On German
  Credit Jev's accuracy falls as its confidence rises.
- The Jev ledger stands at 50,000 of 60,000. Responses are cached by prompt and input, so a
  repeated request costs nothing. Recorded AWS cost is USD 25.67 of a USD 75 limit.
- Learning-curve fractions are `[0.05, 0.1, 0.25, 0.5, 1.0]` in `configs/baseline.yaml`, shared by
  XGBoost and BERT through the same nested-subset function.
- `experiments/thresholding.py`, `configs/costs.yaml` and the escalation stage's validation
  selection implement what is being removed. The paper's main text ends on page 9 of 9.
- The abstract is the user's own text and mentions threshold selection and a selected escalation
  point.

## Goals / Non-Goals

**Goals:**
- Tabular results that are not confounded by class imbalance, with every model on the same
  balanced test rows.
- A prompt-engineering experiment whose every step can be audited and that cannot have used
  test rows.
- A descriptive view of confidence: accuracy and coverage, no chosen threshold.

**Non-Goals:**
- No prompt engineering for Qwen and none on the text datasets.
- No new Qwen job, no recalibration, no new dataset.
- Importance sampling and explanations are not recomputed; they keep the original prompt and
  rows and are labelled so.

## Decisions

### D1. One frozen balanced split per tabular dataset
Undersample the majority class with seed 0 to the minority count, then split each class 60/20/20
with seed 0: German Credit 180/60/60 per class (360 training, 120 validation, 120 test rows);
Breast Cancer Wisconsin 128/42/42 per class (256, 84, 84). `protocol: balanced` in
`configs/datasets.yaml`; the dataset has one "repeat", so stages written for held-out datasets
apply unchanged. The left-out majority rows are saved in the split file and used by nothing.
The new split files replace `german.json` and `wdbc.json`; the manifest lines of the text datasets
stay byte-identical, and `results/splits/REPLACED.json` records the old hashes, the new hashes and
the time.
*Alternative:* repeated balanced splits; rejected because prompt engineering needs rows that are
never test rows, and with repeats every row is a test row somewhere.
*Consequence:* test parts of 120 and 84 rows give bootstrap intervals several points wide; the
test sizes are shown next to every interval.

### D2. Jev with the engineered prompt is its own model entry
Model id `jev_pe`, present for the tabular datasets only, stored in
`results/live/<dataset>/jev_pe/`. `config.MODELS` gains it after `jev`; tables show it as
"Jev, engineered prompt". It is never in `ZERO_SHOT`; its label budget is the training and
validation rows of the balanced split.

### D3. Prompt engineering procedure
A prompt version is a file `configs/prompts/<dataset>/v<k>.yaml` with the instructions, the
option descriptions and serialization settings (feature order, feature subset, display names,
value formatting, an optional short preamble). Option names and the answer format do not change.
- **Round 0** scores the original prompt on the training rows from the saved predictions, at no
  cost, and writes the diagnosis: confusion counts, accuracy and balanced accuracy, accuracy by
  confidence band, and for each feature value the error rate and mean P(positive) against the
  true positive rate.
- **Each later round** (`prompt-engineering round --dataset D --prompt vK --observation "..."
  --change "..."`) sends the training rows with the new version, writes
  `results/live/<dataset>/prompt_engineering/round_K.json` (prompt hash and text, the observation
  and change, results before and after, the same diagnosis tables) and refuses any row outside
  the training part. The analysis between rounds is done by reading these records; no
  observation, no round.
- **At most 8 rounds per dataset.** Up to three versions with the highest training accuracy go to
  validation (`prompt-engineering select`); the best on validation is frozen with its hash, ties
  going to the earlier version; if none beats the original prompt on validation, the original is
  the outcome. `prompt-engineering test` then runs the frozen version on the test part once and
  refuses to run twice.
- **Budget:** stage `prompt_engineering`, 7,000 requests: 8 x (360 + 256) = 4,928 for rounds,
  3 x (120 + 84) = 612 for validation, 204 for test, the rest for retries of failed requests.
*Alternative:* automatic search over prompt variants; rejected because the user asked for changes
motivated by the observed gap.

### D4. Supervised models: 50/75/100% and balanced tabular data
`fractions: [0.5, 0.75, 1.0]`. XGBoost and BERT are run again on all four datasets through the
existing pipelines. For the text datasets the full-data XGBoost predictions are compared with the
saved ones (expected identical); BERT's new run replaces the old one, and any change in its
headline numbers is recorded. Estimated cost USD 2-4; the spend guard applies.

### D5. Accuracy and coverage (`coverage` stage)
Confidence is the maximum class probability; an item is covered when its confidence is at or
above the threshold. Grid: from 1/K to 1.00 in steps of 0.01. Saved per model and dataset:
coverage, accuracy on covered items, covered count and remaining errors. Table thresholds:
0.50, 0.80, 0.90, 0.95, 0.99, 1.00. The table has one row per zero-shot entry and, for XGBoost
and BERT, one row per training size (50%, 75%, 100%), read from the learning-curve probabilities
that the training jobs already save for every size (added 2026-10-02 at the user's request). The
figure shows the 100% models only, to stay legible. Figure: one row per dataset, an accuracy panel and a coverage
panel, one line per model. Accuracy with no covered item is "not defined". Nothing is selected.

### D6. What is removed
The `thresholding` stage, `configs/costs.yaml`, the threshold tables and sweep figures, the cost
columns of the confusion table, and the validation selection in `escalation` (its curve, its
endpoints and the count of unreachable errors stay; for the tabular datasets the curve is on the
balanced test part). The 500 AG News validation predictions bought for the escalation threshold
become unused; they stay in the Jev total and the report says so. The confusion table keeps TP,
FN, FP, TN at the argmax decision for every model entry.

### D7. Metrics and uncertainty
All four datasets are single-split: 95% bootstrap intervals, 1,000 resamples, fixed seed. The
tables show the test size. Mean-and-standard-deviation code paths for repeats are removed.

### D8. Paper
The threshold-selection section is replaced by "Accuracy and coverage" (figure in the main text,
table in an appendix) and a new "Prompt engineering" section (rounds table: observation, change,
training accuracy; final validation and test results for both prompts; the final prompt in an
appendix). The setup describes the balanced splits. The page limit check stays; if the main text
exceeds nine pages, the importance-sampling and explanation tables move back to appendices
first. The abstract is rewritten from the new numbers and shown to the user before the build.

### D9. Safety of existing results
Before anything changes, `results/live/{german,wdbc}`, `results/splits` and the current paper are
copied outside the working copy, and the macros of the text datasets are snapshotted; after the
change only the supervised models' curve and timing macros may differ for the text datasets.

## Risks / Trade-offs

- [Test parts of 120 and 84 rows] → intervals and sizes are shown; conclusions are drawn only
  from differences larger than the intervals.
- [Prompt engineering overfits 360 or 256 training rows] → the choice is made on validation, the
  test part is run once, and training, validation and test results are all reported.
- [The original prompt's aggregate results on all rows were seen before this change] → stated as
  a limitation; the engineered prompt is developed without test rows, enforced by the stage.
- [No prompt improves German Credit] → reported as the outcome, with the rounds tried.
- [BERT's text results shift slightly when rerun] → recorded and stated; the new run is used.
- [Jev budget] → the stage budget and the cap are checked before every round.

## Migration Plan

1. Copy the affected results and snapshot the macros (D9).
2. Balanced splits and freeze; metrics on single splits; remove thresholding (D1, D6, D7).
3. Coverage stage on the saved predictions (D5).
4. Supervised reruns (D4), then prompt engineering rounds, selection and test (D3).
5. Paper, abstract review by the user, build, checks.
Rollback: restore the copied results, split files and paper.

## Open Questions

- The wording of the final abstract, settled with the user after the results are in.
