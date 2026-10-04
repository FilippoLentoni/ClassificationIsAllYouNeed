# Design

## Context

See proposal.md for motivation. Facts established while planning, which shape the approach:

- **Code today:** one dataset (`german`), one backend (`jev`) plus a mock, a sqlite response cache with a Jev ledger inside it, and offline experiment stages that read `results/<mode>/german/jev/predictions.parquet`.
- **What the cache holds:** 99 parsed German Credit responses (probabilities, chosen label, confidence, token count, latency). It does not hold the request or the raw response body. Regenerating the 99 requests from code and hashing them reproduces all 99 stored cache keys, so those requests can be verified but not recovered as wire payloads. The smoke call was never cached.
- **Ledger:** 100 of 100 inferences spent (1 smoke, 99 predict). Pinned model `jev-1.13.0`, `typesafe-sdk` 0.7.2.
- **SDK:** `AsyncTypeSafeClient` accepts a custom `transport` / `http_client`, so the bytes sent and received can be captured.
- **AWS:** the user confirmed the account (`columbia`, <aws-account-id>) and authorised its full use for this work on 2026-09-30. This machine runs in that account as `agentcore-workshop-role` in `us-east-2`. The account is not in an AWS Organization, so no organisation-level policy can block SageMaker. The role is currently denied SageMaker and Service Quotas actions, there is no SageMaker execution role and no S3 bucket. The role has IAM, S3 and CloudFormation full access, so it can grant itself what the baseline needs. The SageMaker XGBoost container `3.0-5` is available in `us-east-2`. `boto3`, `sagemaker`, `datasets` and `xgboost` are not installed.
- **Machine:** 2 CPUs, 3.8 GB RAM, so the baseline's feature building and training run in SageMaker, not locally.
- **Repository:** `git init` has been run, nothing is committed, and `.gitignore` excludes `cache/` and the live predictions file.
- **Prior public evaluations** (from web search; numbers not yet checked against the sources): Deußer, Sparrenberg and Sifa (arXiv:2609.37647, `jev-1.13.0`) report SST-2 validation (872) accuracy 0.964 / ECE 0.015, AG News test (7,600) 0.885 / 0.077, Banking77 test (3,076) 0.797 / 0.087, with one frozen template per dataset tried on 20 training or validation examples. `dhruvmehra/jevbench` covers the same three on 500-item samples. Nothing was found for German Credit. For Breast Cancer Wisconsin Diagnostic there is only a second-hand mention.

## Goals / Non-Goals

**Goals:**
- Every Jev request from now on is logged exactly as sent and received, before any new request is made.
- Evaluation identifiers are frozen and recorded before any template check, baseline training or threshold selection.
- Jev and XGBoost are scored on identical examples, and evaluation labels never leave this machine.
- No measurement enters the paper unless it is in a saved output; the historical 100 calls are described as what they are.
- The whole study stays within 60,000 Jev inferences and a USD 5 AWS spending limit.

**Non-Goals:**
- Recalibrating Jev or XGBoost. Both are reported as returned; the calibration label budget is zero for both and is stated as such.
- Explanations for XGBoost (TreeSHAP). Perturbation experiments are on Jev only.
- Importance sampling for the baseline. It is run on Jev's cached predictions only.
- Prompt optimisation. One template per dataset, frozen after at most one training-split check.
- Additional baselines beyond XGBoost.
- A silent local substitute for SageMaker. If SageMaker cannot run, the baseline is reported as incomplete.

## Decisions

### D1. Datasets, sources and splits

| Dataset | Source | Classes (positive) | Evaluation examples | Training / validation |
|---|---|---|---|---|
| German Credit | OpenML `credit-g` (id 31) | good, bad (bad) | 10 stratified repeats, test 200 each, over all 1,000 rows | per repeat: train 600, val 200 |
| Breast Cancer Wisconsin Diagnostic | scikit-learn bundled copy of UCI WDBC (569 rows, 30 features) | benign, malignant (malignant) | 10 stratified repeats, test 114 each, over all 569 rows | per repeat: train 341, val 114 |
| SST-2 | GLUE `sst2` on Hugging Face | negative, positive (positive) | official validation, 872 | official train 67,349: val 2,000, train 65,349 |
| AG News | `fancyzhx/ag_news` | World, Sports, Business, Sci/Tech | official test, 7,600 | official train 120,000: val 5,000, train 115,000 |
| Banking77 | PolyAI's GitHub repository `PolyAI-LDN/task-specific-datasets`, pinned commit | 77 intents | official test, 3,080 | official train 10,003: val 1,000, train 9,003 |

- Repeats use seeds 0–9 with a 60/20/20 stratified split. Validation parts are stratified draws with seed 0.
- **Jev is zero-shot, so it predicts every tabular row once** (1,000 and 569). Each repeat then scores Jev on that repeat's test identifiers, which are exactly the rows XGBoost is scored on, and selects its threshold on that repeat's validation identifiers. The repeats are random draws, so not every row is a test row: 885 of 1,000 German Credit rows and 506 of 569 breast-cancer rows are in at least one test part. *Alternative:* querying Jev only on one test split; rejected because repeated splits would then apply to the baseline only.
- SST-2's official test split is unlabelled, so its labelled validation split is the held-out set, as in the prior evaluation.
- Hugging Face datasets are pinned to a revision hash recorded in the dataset documentation. Banking77 is read from the two CSV files in PolyAI's GitHub repository at a pinned commit with recorded checksums, because the Hugging Face `PolyAI/banking77` repository is a loading script that the current `datasets` library refuses; those are the files that script downloads. The `mteb/banking77` copy has 3,076 test items, which is where the prior paper's count comes from. *Alternative:* scikit-learn / OpenML mirrors for the text datasets; rejected because the official splits are what prior work used.
- The old German Credit split (pools of 200/300, subset 39/60) is deleted. Its 99 cached responses are reused automatically, because the template and therefore the cache keys are unchanged.

### D2. Templates and freezing

- One Choice question per example. `instructions` is a one-sentence task description; `criteria` maps each label to a description, in the dataset's class order. German Credit keeps its existing template unchanged.
- Banking77 descriptions follow one mechanical rule: "The customer's message is about: <intent name with underscores replaced by spaces>." *Alternative:* hand-written definitions for 77 intents; rejected as untestable prompt design.
- **Freeze order:** `freeze` writes `results/splits/<dataset>.json` and `results/splits/MANIFEST.sha256`, then the dataset documentation. Only afterwards may anything else run; every stage checks the manifest hash first.
- **Template check:** at most 20 training-split examples for each text dataset (60 calls in total). The tabular datasets get no check on data rows, because every row is an evaluation row in some repeat; their response shape is validated by the first request of `predict`, which stops the stage if the shape is wrong. Any template change after a check is recorded with its reason, and templates are frozen in `results/live/templates/<dataset>.json` before `predict`.

### D3. Jev budget

`jev_total: 60000`, enforced by the existing ledger, which already holds 100.

| Stage | Budget | Use |
|---|---|---|
| smoke | 5 | 1 already spent |
| template | 60 | 20 per text dataset |
| predict | 13,700 | 13,621 predictions (99 already spent), rest for re-running failed requests |
| explain | 44,000 | 5 datasets × 20 examples × at most 423 requests |
| unallocated | 2,235 | not usable without editing the budget file |

Estimated input volume is about 35 million tokens, about USD 1.50 at USD 0.042 per million. Banking77 requests are the largest because each carries 77 label descriptions.

### D4. Request and response log

- A wrapping HTTP transport passed to the SDK client records the request body and the response body of every call as they cross the wire. Headers are not recorded, so the key cannot enter the log. *Alternative:* logging the arguments passed to the SDK; rejected because it would show what we asked the SDK to send, not what was sent.
- Each record also carries the ledger sequence number, stage, dataset, cache key, SDK version, requested model, timestamp and latency, so it can be joined to predictions.
- Records are appended to gzip JSONL files under `cache/exchanges/` (expected about 100 MB raw, too large and too noisy for git).
- `results/live/payload_examples/<dataset>.json` (tracked) holds the five exchanges shown in the paper, each copied from a log record of a seeded evaluation example.
- **Historical calls:** no records are written for them. A `payload-audit` command regenerates the 99 requests, reports how many hashes match the cache, and writes that result to a tracked file. The paper cites that result and states that the smoke call and the 99 raw responses were not recorded.

### D5. Saved artifacts and backup

- `.gitignore` stops excluding `results/live/**/*.parquet`; predictions for every dataset and model are tracked.
- After each live stage, `cache/responses.sqlite` (cache and ledger) and `cache/exchanges/` are copied to the project bucket under `backup/<timestamp>/`. Until the bucket exists, the copy goes to a local directory outside the working copy and the stage says so.
- The explain stage cannot be re-run without the cache; its result files are tracked, and the paper builds from them.

### D6. Supervised baseline on SageMaker

**Resources** (`infra/sagemaker.yaml`, one CloudFormation stack `ciayn-baseline` in `us-east-2`):
- a private S3 bucket (public access blocked, encrypted, 30-day expiry on job inputs and outputs);
- an execution role `ciayn-sagemaker-exec` trusted by SageMaker, limited to that bucket, its own log groups and pulling the SageMaker images;
- a managed policy attached to `agentcore-workshop-role` allowing SageMaker pipeline, processing-job and training-job actions on resources named `ciayn-*`, passing only the execution role, and reading SageMaker service quotas (needed for the quota check, which the role cannot do today).

The template and the exact deploy command are shown to the user and need approval before anything is created. Teardown empties the bucket and deletes the stack. *Alternative:* creating resources with individual CLI calls; rejected because a single template is reviewable and removable as a unit.

**Pipeline** (`ciayn-xgb`, parameterised by dataset). The pipeline definition is built as plain data by our own code and submitted with `boto3`. *Alternative:* the SageMaker Python SDK; rejected because it uploads code to S3 while building a definition, so the definition could not be generated or tested offline, and it is a heavy dependency for two steps.
1. **Processing job** (scikit-learn container): reads train and validation examples with labels and evaluation examples *without labels*; fits preprocessing on train only, per repeat; writes sparse feature matrices. Tabular: one-hot encoding of categoricals, numeric features unchanged. Text: TF-IDF on unigrams and bigrams, at most 50,000 features.
2. **Training job** (XGBoost container `3.0-5`, script mode): for each repeat, a fixed grid of `max_depth` {3, 6} × `eta` {0.1, 0.3} with early stopping on validation (at most 500 rounds, patience 30), selected by validation log-loss; then class probabilities for validation and evaluation examples; then the learning curve at training fractions {0.05, 0.1, 0.25, 0.5, 1.0} with the selected hyperparameters. It writes predictions, learning-curve probabilities and a run record (hyperparameters, best iterations, label counts, library versions).

- One training job per dataset loops over repeats, grid and curve sizes. *Alternative:* one job per fit; rejected because about 300 job start-ups would cost far more time and money than the fits.
- **Test labels are withheld from every fit.** For the held-out datasets, evaluation labels never leave this machine. For the repeated-split datasets a row that is a test row in one repeat is a training row in another, so its label is uploaded, but each repeat is packaged separately and its test rows are supplied without labels. All metrics, including the learning curve, are computed locally from the returned probabilities.
- **Finding a combination that runs (user decision, 2026-09-30):** if the preferred instance type or region is refused, iterate without asking until one works. Order: instance types `ml.m5.xlarge`, `ml.m5.large`, `ml.m5.2xlarge`, `ml.c5.xlarge`, `ml.m4.xlarge`, `ml.c5.2xlarge` in `us-east-2`; then the same list in `us-east-1`, `us-west-2` and `eu-west-1`. For each candidate the quota is read where possible, and otherwise the German Credit execution (the smallest) is started as the probe; a refused job costs nothing. Every attempt is written to `results/live/baseline_attempts.json`. The first combination that completes the German Credit execution is used for all five datasets, so every dataset runs on the same hardware and container. The spending limit applies across all attempts. Only public dataset text and features are uploaded, so the region does not matter for the data.
- The template is approved once; deploying the same template in another region while iterating needs no further approval. Role and policy names carry the region, because IAM names are account-wide.
- Preferred instance type `ml.m5.xlarge` for both jobs; stopping condition 1 hour for processing and 3 hours for training.
- **Text datasets use 16 histogram bins** instead of XGBoost's default 256. The first AG News run at the default was stopped after 30 minutes when a local timing showed the full grid would exceed the limit; TF-IDF weights are sparse, so coarse bins cost nothing measurable on validation log-loss and make each round several times faster. The setting was chosen from run time and validation loss only, applies to all three text datasets (SST-2 was rerun with it), and is reported in the paper. Learning-curve fits are skipped, and recorded as skipped, once 80% of the training time limit is used. Estimated total about USD 1; `aws_spend_limit_usd: 5` is checked before each execution using instance price × maximum runtime for the jobs about to start plus billable seconds already recorded.
- Pipeline execution identifiers, job names, container versions and billable seconds are saved in `results/live/<dataset>/xgb/run.json`.
- In mock mode the baseline is a second synthetic predictor ("mock-xgb"); tests never touch AWS.
- `xgboost` is added to the development dependency group only, so the training script can be unit-tested on a toy dataset. It is never used locally to produce a result.

### D7. Metrics and uncertainty

- Binary: ROC-AUC and PR-AUC for the positive class. Multiclass: one-vs-rest macro averages over classes present.
- Held-out datasets: bootstrap 95% intervals, 1,000 resamples, seed 0. Repeated-split datasets: mean and standard deviation over the 10 repeats.
- ECE keeps 15 equal-width bins.
- Banking77's 77-row per-class table goes to a tracked CSV and the appendix.

### D8. Threshold selection

- Rule: positive when P(positive | x) ≥ t, swept over t ∈ {0, 0.01, …, 1}.
- Validation-selected t minimises mean decision cost on validation examples; ties go to the threshold closest to the analytic one, then to the smaller.
- Cost matrices: German Credit canonical 5:1. Breast Cancer Wisconsin Diagnostic 5:1 for a missed malignant case, labelled illustrative. SST-2 1:1, so its analytic threshold equals the default and the figure says so.
- Jev's threshold is selected on validation rows it has already predicted: each repeat's validation part for the tabular datasets, and the first 500 of SST-2's validation draw (500 extra Jev calls, included in D3). XGBoost uses its full validation part.
- Figures show repeat 0 for the repeated-split datasets; tables give mean and standard deviation over repeats.
- The genetic algorithm, its configuration and its tests are deleted.

### D9. Importance sampling

- Pool per dataset: all evaluation examples with cached Jev predictions (1,000; 569; 872; 7,600; 3,080).
- The sampler is a function of the prediction matrix and a label oracle. The oracle reveals a label only for a drawn index and counts its calls; the experiment asserts the count equals the draws.
- Proposal q ∝ (1 − ε)(1 − p_max)^α + ε/N with ε = 0.1, α = 1, plus α ∈ {0.5, 2} as sensitivity. Sampling with replacement; Hansen–Hurwitz estimate of the error rate, accuracy as one minus it; CLT intervals.
- Budgets {25, 50, 100, 200, 400}, 1,000 seeded trials each. The macro-F1 estimator is deleted.

### D10. Live explanations

- 20 seeded evaluation examples per dataset. Text examples are drawn among those with 6 to 40 whitespace-separated words, so that k = 5 deletion is meaningful and the perturbation count is bounded; the rule is stated in the paper.
- Features are the 20 or 30 columns (tabular) or the words (text); masking removes them from the input.
- KernelSHAP with 200 coalitions and the all-removed input as background; LIME with 200 samples; deletion at k ∈ {1, 3, 5} against 5 random draws. At most 423 requests per example.
- The stage computes its worst-case request count and checks it against D3 before sending anything, then runs German Credit first as a sanity check before the other four.
- The mock probability function is generalised to text and many classes and is kept for tests only.

### D11. Paper, prior work and verification

- Sections as in the `paper-manuscript` spec. The triage section, its figures and macros are deleted.
- `configs/prior_work.yaml` holds each cited prior number with source, URL, dataset, split, sample size, quoted text, date checked and protocol notes. The generator refuses an entry without a check date. No prior number is typed into the prose.
- The appendix is generated from `results/live/payload_examples/` and the dataset documentation.
- `verify-paper` regenerates every macro from the saved outputs into a temporary directory and compares it with `paper/generated/`, then confirms that each macro value used in the text appears in the text extracted from the PDF, and writes `results/live/paper_check.json`.
- A page-by-page inspection of the rendered PDF is recorded in `docs/PDF_INSPECTION.md`.
- `docs/REPORT.md` gives findings, limitations, actual costs (Jev from the ledger and logged token counts; AWS from billable seconds) and incomplete items.
- `make paper` builds in live mode; `make paper MODE=mock` builds the watermarked draft.

### D12. Removal

`experiments/triage.py`, `configs/triage.yaml`, the genetic algorithm in `experiments/thresholding.py`, their tests, the triage paper section, and the corresponding Makefile targets and CLI stage.

## Risks / Trade-offs

- **The account may have zero quota for `ml.m5.xlarge` processing or training jobs.** The quota could not be read during planning because the role lacks that permission. → The template adds read access to quotas, and the stage iterates over the configured instance types and regions until one runs. If all 24 combinations are refused, the stage stops, reports every attempt, and the paper reports the baseline as incomplete.
- **Iterating across regions leaves resources in more than one region.** → Each deployed stack is recorded, stacks in regions that were not used are deleted as soon as a working combination is found, and teardown covers every recorded region.
- **Granting permissions to `agentcore-workshop-role` changes a role other work on this account may use.** → The user has authorised full use of the account; the policy is still scoped to `ciayn-*` resources, is part of the reviewed template, and is removed by teardown.
- **Teardown deletes the bucket that holds the cache backup.** → Teardown refuses to run unless the local cache, ledger and exchange log exist and match the latest backup.
- **Jev calls cannot be unspent.** → Logging and freezing are implemented and tested in mock mode before the first new call; every stage prints its dry-run count first; failed requests are counted and only re-run within budget.
- **`jev-1.13.0` may no longer be served, or the API may be rate-limited.** → The smoke check runs first; if the reported model differs from the pinned one, the run stops and the user decides.
- **Banking77 requests are large and the 77-way explanation is expensive.** → Its explain share is about 8,500 requests, inside the budget; German Credit runs first so a problem shows before the large spend.
- **The prior paper reports 3,076 Banking77 test examples and the official split has 3,080.** → We evaluate the full official split and state the difference as a protocol note.
- **Public numbers were read through a web summariser during planning.** → Each is checked against the source before it enters `configs/prior_work.yaml`; an unverifiable number is dropped.
- **Zero-shot against supervised is an unequal comparison.** → Label budgets are reported for both and the learning curve shows how many labels the baseline needs to match Jev.
- **Selecting text examples of 6–40 words biases the explanation sample.** → The rule and the share of examples it covers are reported.
- **The paper will exceed the earlier 6–8 pages.** → The body target becomes 8–10 pages, with payloads, dataset documentation and per-class tables in the appendix.

## Migration Plan

1. With the user's approval, commit the current working tree so the 100-inference version stays recoverable.
2. Implement and test everything in mock mode for all five datasets; no spend.
3. Freeze identifiers and dataset documentation; commit them.
4. Turn on request logging, run the smoke check, template checks and predictions, backing up after each.
5. Review and deploy the AWS template; check quota; run the five pipeline executions; fetch results.
6. Run the offline experiments, then live explanations (German Credit first).
7. Verify prior-work numbers, build the paper, run `verify-paper`, inspect the PDF, write the report.
8. Tear down AWS resources when the user confirms the local copies are safe.

Rollback: AWS resources are removed by teardown; code and results return to the step-1 commit. Spent Jev inferences and AWS charges cannot be undone.

## Open Questions

- The exact Hugging Face revision hashes, recorded when the datasets are first downloaded.
- Which instance type and region end up running the jobs; settled by the iteration in D6 and recorded with the results.
- The primary source of the exploratory tabular benchmark mentioning Breast Cancer Wisconsin Diagnostic; if it cannot be located, the paper says no verifiable prior evaluation was found.
