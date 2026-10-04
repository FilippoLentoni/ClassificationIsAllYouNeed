# Tasks

## 1. Preserve the current state

- [ ] 1.1 Copy `results/live/german`, `results/live/wdbc`, `results/splits`, `paper/` and `paper/generated/macros.tex` outside the working copy; verify the copies exist and list their sizes.

## 2. Balanced tabular splits and single-split evaluation

- [ ] 2.1 Add `protocol: balanced` (undersample to equal classes, one 60/20/20 split per class, fixed seeds, left-out rows recorded) for German Credit and Breast Cancer Wisconsin; verify with tests that each part has equal class counts (180/60/60 and 128/42/42 per class), parts are disjoint, two loads agree and left-out rows are used by nothing.
- [ ] 2.2 Replace the two tabular split files in the manifest without changing the text datasets' lines and write `results/splits/REPLACED.json`; verify the SST-2 and AG News files and hashes are byte-identical and every stage passes the manifest check.
- [ ] 2.3 Make the metrics, comparison, importance and explanation stages read the tabular datasets as single-split with bootstrap intervals, and remove the repeat-averaging paths; verify with tests and that the text datasets' metrics are unchanged against the snapshot.
- [ ] 2.4 Remove the `thresholding` stage, `configs/costs.yaml`, its tests, tables and figures, and the validation selection from `escalation` (curve, endpoints and unreachable-error count stay); verify with tests that no result file or generated file contains a selected threshold or a decision cost.
- [ ] 2.5 Update the dataset documents and `docs/` for the balanced splits; verify the documented counts equal the split files.

## 3. Accuracy and coverage

- [ ] 3.1 Add the `coverage` stage (grid from 1/K to 1.00, coverage, accuracy on covered items, remaining errors, table thresholds with a row for XGBoost and BERT at each training size, figure with an accuracy and a coverage panel per dataset and one line per model); verify with tests the lowest-threshold case, the empty-coverage case and that the Jev ledger and the SageMaker record are unchanged.
- [ ] 3.2 Run it on the saved predictions for Jev and Qwen and show the table for all available models to the user; verify the values for Jev on the text datasets equal the exploratory table already shown.

## 4. Supervised models

- [ ] 4.1 Set the learning-curve fractions to 50%, 75% and 100% for XGBoost and BERT and update packaging and tests; verify with a test that the smallest size is half the training part and both models use the same subsets.
- [ ] 4.2 Run and fetch XGBoost on all four datasets; verify identifiers match, the text datasets' full-data predictions equal the saved ones, and the recorded cost is below the limit.
- [ ] 4.3 Run and fetch BERT on all four datasets; verify identifiers match and record any change in its text-dataset headline numbers.
- [ ] 4.4 Rerun the `coverage` stage with the new learning-curve probabilities; verify the table has XGBoost and BERT rows at 50%, 75% and 100% for every dataset and that each 100% row equals the model's main row.

## 5. Prompt engineering for Jev

- [ ] 5.1 Add prompt versions as files (instructions, option descriptions, serialization settings) and the `jev_pe` model entry; verify with tests that version 0 reproduces the original request byte for byte and that option names cannot change.
- [ ] 5.2 Add `prompt-engineering round | select | test` with the stage budget of 7,000, the training-rows-only guard, the per-round record (prompt, observation, change, before and after, diagnosis tables) and the run-once test guard; verify with tests on the fake Jev server that a non-training row is refused in a round, that a round without an observation is refused, that the budget is enforced, and that `test` refuses a second run or a changed prompt.
- [ ] 5.3 Round 0 for both datasets from the saved predictions (no requests); verify the records exist and show them to the user.
- [ ] 5.4 Run the rounds for German Credit (at most 8), each motivated by the previous record; verify every round has an observation and a change and the ledger grew only by training-row requests.
- [ ] 5.5 Run the rounds for Breast Cancer Wisconsin (at most 8); verify as in 5.4.
- [ ] 5.6 Select on validation, freeze, and run the test parts once; verify the frozen hash, the saved `jev_pe` predictions and that the stage total is within 7,000.

## 6. Results, paper and report

- [ ] 6.1 Run metrics, comparison, coverage, escalation and provenance in live mode; verify every model entry is scored on identical test identifiers and the text datasets' zero-shot numbers equal the snapshot.
- [ ] 6.2 Generate the new tables and figures (five model entries on the tabular datasets, label table with prompt-engineering labels, confusion counts without costs, coverage figure and table, prompt-engineering rounds table, final prompt appendix); verify the mock pipeline builds them too.
- [ ] 6.3 Rewrite the sections: setup (balanced splits), accuracy and coverage in place of threshold selection, the prompt-engineering section, escalation as a curve, limitations; verify every number is a generated macro and no threshold-selection text remains.
- [ ] 6.4 Draft the abstract from the new numbers and show it to the user with its numbers; build the final paper only with the text the user confirms.
- [ ] 6.5 Build and run `verify-paper`; verify numbers and all format checks pass and the main text is within nine pages, moving the importance-sampling and explanation tables to appendices first if needed.
- [ ] 6.6 Inspect every page, update `docs/PDF_INSPECTION.md`, regenerate `docs/REPORT.md` and the README; verify the fresh-copy check passes with zero Jev requests and zero SageMaker jobs, and back up the new results.
