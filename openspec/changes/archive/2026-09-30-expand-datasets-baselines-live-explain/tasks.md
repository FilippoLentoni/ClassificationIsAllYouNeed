# Tasks

## 1. Baseline commit, dependencies and removals

- [ ] 1.1 With the user's approval, commit the current working tree as the pre-expansion version, and stop ignoring `results/live/**/*.parquet` in `.gitignore`; verify `git log` shows the commit and `git check-ignore results/live/german/jev/predictions.parquet` prints nothing
- [x] 1.2 Add `datasets` and `boto3` to the dependencies and `xgboost` to the development group only; verify `uv sync` succeeds, the three packages import, and the existing test suite still passes
- [x] 1.3 Remove the triage experiment and the genetic algorithm (design D12): stage modules, configs, tests, CLI stage, Makefile targets, and the triage section, figures and macros of the paper; verify `grep -ri "triage\|genetic" src tests configs Makefile paper/sections paper/main.tex` finds nothing, the test suite passes and `make paper MODE=mock` still builds

## 2. Datasets and frozen identifiers

- [x] 2.1 Generalise the dataset model (tabular or text, ordered classes, positive class for binary, split repeats) and add loaders for Breast Cancer Wisconsin Diagnostic, SST-2, AG News and Banking77 with pinned sources (design D1); verify with tests that the evaluation sets have 872, 7,600 and 3,080 examples and the tabular datasets 1,000 and 569 rows, with the documented class order
- [x] 2.2 Implement the 10 seeded stratified 60/20/20 repeats for the tabular datasets and the seeded validation draws from the official training splits for the text datasets, and delete the old German Credit split and subset; verify with tests that repeats are identical across two loads, disjoint within a repeat, stratified, and that no text validation or training identifier is in the held-out split
- [x] 2.3 Extend serialization to the 30 numeric breast-cancer features and to text inputs with the documented normalisation; verify with tests that every tabular feature appears exactly once as `name: value`, missing values read "unknown", and no serialization contains the gold label
- [x] 2.4 Implement the `freeze` stage (split files, `MANIFEST.sha256`) and a manifest check that every other stage runs first; verify with a test that altering a split file makes a stage stop before doing work and name the mismatch
- [x] 2.5 Generate the dataset documentation for each dataset (source and revision, classes and order, positive class, split sizes and repeats, preprocessing and missing-value handling, seeds) from the experiment configuration; verify with a test that each of the five files contains every listed item
- [ ] 2.6 Run `freeze` and, with the user's approval, commit the split files, the manifest and the dataset documentation before any template check, baseline run or threshold selection; verify the commit contains `results/splits/MANIFEST.sha256` and its hash matches the files

## 3. Harness: budget, logging, mock and persistence

- [x] 3.1 Raise the cap to `jev_total: 60000` with stage budgets smoke 5, template 60, predict 13,700, explain 44,000 (design D3), and make every remote stage report and stop on `--dry-run`; verify with tests that a ledger pre-filled with 100 entries counts toward the cap, that a stage exceeding its budget or the cap stops before any call, and that a dry run leaves the ledger unchanged
- [x] 3.2 Implement request and response logging at the HTTP transport (design D4): request body as sent, response body as received, SDK version, requested and reported model, ledger number, stage, dataset, cache key, timestamp, latency; verify with tests on a stub transport that one record is written per request, that a failed request is recorded, and that neither the key nor any header appears in the log
- [x] 3.3 Implement `payload-audit` for the 100 historical calls: regenerate the 99 cached requests, compare their hashes with the stored cache keys, and write the result to a tracked file that also states that the smoke call and the raw responses were not recorded; verify it reports 99 of 99 matches and adds no record to the exchange log
- [x] 3.4 Generalise the mock to both models, all five datasets, text inputs and 77 classes, including the masking-based probability function for words; verify with tests that outputs are deterministic and order-independent, mock-labelled, and produced with no API key and no AWS credentials
- [x] 3.5 Rework `predict` and the results layout for every (dataset, model): Jev predicts all tabular rows, the text evaluation sets and the 500 SST-2 validation items, and the first request's response shape is checked before the rest are sent; verify with stub-SDK tests and with `make predict MODE=mock` writing predictions for all five datasets and both models, with zero calls on a second run
- [x] 3.6 Implement the `template-check` stage (at most 20 training-split examples per text dataset) and the frozen template files; verify with a test that every example used is a training identifier and that `predict` refuses to run for a dataset whose template file is missing
- [x] 3.7 Implement the backup of the cache, ledger and exchange log after every live stage, to the project bucket or to a local directory outside the working copy when the bucket does not exist; verify with a test using a temporary directory that the copies are identical and the location and time are reported
- [x] 3.8 Update `docs/JEV_ACCESS.md` for the new cap, stage budgets, request logging, backup and estimated cost; verify every command in it runs as written in mock or dry-run mode

## 4. Supervised baseline on SageMaker

- [x] 4.1 Write `infra/sagemaker.yaml`, deployable in any configured region with region-specific names (bucket, execution role, scoped policy for `agentcore-workshop-role` including read access to SageMaker quotas) and `docs/AWS_BASELINE.md` (what is created, deploy command, cost estimate, spending limit, teardown); verify `aws cloudformation validate-template` accepts the template
- [x] 4.2 Write the processing script: fit one-hot encoding or TF-IDF on training examples only, per repeat, and transform validation and evaluation examples; verify with tests on toy data that a category or word seen only outside training has no fitted entry and that outputs cover every repeat
- [x] 4.3 Write the training script: the fixed hyperparameter grid with early stopping on validation, selection by validation log-loss, probabilities for validation and evaluation examples, the learning-curve fits, and the run record with label counts; verify with a test on a toy dataset that predictions exist for every repeat and curve size and that the record holds training, validation and threshold-selection label counts
- [x] 4.4 Write the pipeline definition, the input packaging and the spending-limit guard; verify with tests that the definition is identical across two generations, that the packaged evaluation examples carry no labels, and that the guard blocks a run whose estimate exceeds `aws_spend_limit_usd`
- [x] 4.5 Add the `baseline plan|deploy|run|fetch|teardown` commands, with `fetch` refusing predictions whose identifiers differ from the frozen evaluation identifiers and `teardown` refusing to run unless local copies match the latest backup; and `run` iterating over the configured instance types and regions (design D6) and recording every attempt; verify with tests for the identifier mismatch and the teardown refusal, a test with a stubbed SageMaker client that a refused instance type moves on to the next and that exhausting all combinations stops with every attempt reported, and that `baseline run --dry-run` lists the jobs without starting any
- [x] 4.6 Show the user the template and the deploy command, and deploy only after approval; verify the stack reaches `CREATE_COMPLETE` and `aws sagemaker list-pipelines --region us-east-2` succeeds
- [x] 4.7 Run the German Credit pipeline execution as the probe, iterating over instance types and then regions without asking until one completes (deploying the approved template in a new region when needed, and deleting stacks in regions that were not used), and fetch its results; verify the execution succeeded, `results/live/baseline_attempts.json` lists every attempt, `run.json` records the instance type, region, execution identifier, container versions and billable seconds, and the prediction identifiers equal the frozen ones. Stop and report to the user only if every combination is refused
- [x] 4.8 Run and fetch the remaining four pipeline executions on the same instance type and region; verify baseline predictions and learning-curve outputs exist for all five datasets and the recorded AWS cost is within the spending limit

## 5. Live Jev predictions

- [x] 5.1 Run the smoke check with logging on; verify the reported model is `jev-1.13.0` (stop and ask the user if not), the exchange is in the log, and the log contains no key
- [x] 5.2 Run the template checks and freeze the templates; verify the ledger shows at most 60 template requests, the examples used are training identifiers, and a template file exists for each dataset
- [x] 5.3 Run `predict` in live mode after reviewing its dry-run report, tabular datasets first; verify predictions are saved and tracked for all five datasets, invalid and missing counts are reported, the ledger is within the predict budget, the backup ran, and a second run makes zero requests
- [x] 5.4 Export the five payload examples from the log; verify with a test that each exported exchange is identical to its log record and that the Banking77 example holds all 77 label descriptions and probabilities

## 6. Metrics and calibration

- [x] 6.1 Extend the metrics stage to every dataset and model, with one-vs-rest macro ROC-AUC and PR-AUC for multiclass and a per-class CSV, and stop if Jev and baseline identifiers differ; verify with tests against hand-computed values on a toy three-class set and a test for the identifier mismatch
- [x] 6.2 Report uncertainty as bootstrap intervals for the held-out datasets and as mean and standard deviation over repeats for the tabular datasets; verify with tests on synthetic predictions for both cases
- [x] 6.3 Produce calibration numbers and a reliability diagram for every pair, the learning-curve figure with Jev's zero-shot result as a reference line, and the label-budget table; verify `make metrics MODE=mock` writes them for all five datasets and that the table shows zero training labels for Jev

## 7. Threshold selection

- [x] 7.1 Implement the rule "positive when P(positive | x) ≥ t", the sweep, the validation-selected threshold with its tie-break, and the three-threshold table per binary dataset and model (design D8); verify with tests for the boundary case, deterministic selection, selection unchanged when evaluation labels are permuted, and mean and standard deviation over repeats
- [x] 7.2 Add the cost matrices with their canonical or illustrative flag and the two-panel figure (metric curves; decision cost) with the three thresholds marked and macro-F1 as a separately labelled curve; verify `make thresholding MODE=mock` writes one figure per binary dataset and model and the results mark the breast-cancer costs as illustrative

## 8. Importance sampling

- [x] 8.1 Rewrite the estimator as accuracy-only with a label oracle (design D9); verify with tests that the oracle is called only for drawn items, the proposal is unchanged when labels are permuted, every sampling probability is positive including for items predicted with probability 1, and the estimate is unbiased on a synthetic pool within Monte Carlo error
- [x] 8.2 Run the stage on the five Jev pools at budgets {25, 50, 100, 200, 400} with the table and one figure per dataset; verify `make importance MODE=mock` completes with zero remote calls and writes them

## 9. Live explanations

- [x] 9.1 Generalise the explanation stage to text words and to the live Jev backend, with the example-selection rule and a worst-case request count checked against the budget before anything is sent; verify with tests on the mock that every word gets a LIME weight, additivity error is reported, and the stage stops before any request when the count exceeds the budget
- [x] 9.2 Run live explanations for German Credit after reviewing the dry-run count; verify the results are live-labelled, the additivity error and deletion results are present, and the requests used are within 423 per example
- [x] 9.3 Run live explanations for the other four datasets; verify results exist for all five, the ledger's explain total is within 44,000, and the backup ran

## 10. Paper

- [x] 10.1 Check every prior-work number against its source and record it in `configs/prior_work.yaml` with source, quoted text, date checked and protocol notes, dropping any that cannot be verified; verify with a test that the generator refuses an entry without a check date and that the paper states where no prior evaluation was found
- [x] 10.2 Rework the paper generator: tables for metrics by dataset and model, label budgets, thresholds, importance sampling, explanations and prior work; the appendix with the five exchanges, dataset documentation and per-class tables; costs computed over all inferences in the ledger plus the AWS cost; verify with tests that generation fails when a dataset has no logged exchange, that the mock flag is set for mock inputs, and that the reported inference count equals the ledger total
- [x] 10.3 Rewrite the prose for the new structure (related evaluations, setup, four results sections, limitations, conclusion), labelling Jev as zero-shot and the baseline as supervised and stating which requests predate logging; verify `make paper MODE=mock` builds with the watermark and no LaTeX errors and that no result number is typed by hand
- [x] 10.4 Implement `verify-paper` (design D11); verify with a test that a deliberately altered generated number makes it fail and name the mismatch

## 11. Integration and delivery

- [x] 11.1 Run the offline stages and `make paper` in live mode, then `verify-paper`; verify it confirms every reported number against the saved outputs and that no mock artifact was consumed
- [x] 11.2 Inspect every page of the rendered PDF for missing figures, overflowing tables and unresolved references, fix what is found, and record the inspection in `docs/PDF_INSPECTION.md`; verify the record lists each page
- [x] 11.3 Write `docs/REPORT.md` (findings, limitations, actual Jev and AWS costs, anything incomplete) and a `README.md` with the reproducible commands; verify each command runs as written in mock mode and each figure in the report matches the saved outputs
- [x] 11.4 Check a fresh checkout without the response cache; verify the metrics, thresholding and importance stages and the paper build complete from tracked files with zero Jev requests and zero SageMaker jobs
- [ ] 11.5 When the user confirms the local copies are safe, tear down the AWS resources; verify the stack is deleted in every region where it was deployed and no `ciayn-*` bucket, role or policy remains
