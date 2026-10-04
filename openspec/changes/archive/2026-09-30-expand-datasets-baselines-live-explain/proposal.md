# Proposal

## Why

The current paper rests on 60 German Credit test items, one model and no baseline, because the whole study was capped at 100 Jev inferences. That is too little to support its claims: intervals are wide, there is nothing to compare Jev against, explanations were never run on Jev, and the reader cannot see what Jev was actually sent. Public evaluations of the same Jev version now exist on standard text benchmarks, so the paper can be placed next to them and extended to the tabular setting they do not cover.

## What Changes

- **Datasets: one becomes five.** Keep German Credit and add Breast Cancer Wisconsin Diagnostic (binary tabular), SST-2 (binary sentiment), AG News (4-class text) and Banking77 (77-class intent). Official held-out splits are used where they exist; the two small tabular datasets use repeated stratified splits. Evaluation IDs are frozen and recorded before any tuning. Dataset versions, class definitions, sample sizes, preprocessing and seeds are documented.
- **BREAKING: the 100-inference cap is replaced by a 60,000-inference cap.** The persistent ledger stays and keeps counting from the 100 already spent. Estimated Jev cost is under USD 3.
- **Exact request and response logging.** Every new Jev call records the request as sent and the response as received. The paper shows five complete exchanges, one per dataset, taken verbatim from that log. The first 100 calls predate logging: their requests can be regenerated and checked against the stored cache keys (99 of 99 match today), but their raw responses were never stored and the smoke call was not cached at all. They are never presented as original payloads.
- **BREAKING: a supervised baseline returns.** XGBoost is trained and scored through an Amazon SageMaker pipeline (processing and training jobs) on the identical test examples. Preprocessing is fitted on training data only, tuning never sees test labels, and all training, calibration and threshold-selection label budgets are reported. An XGBoost learning curve is included if feasible. This needs new AWS resources (one S3 bucket, one SageMaker execution role, SageMaker permissions for this machine's role), created from a single reviewable template and removable with one command.
- **Metrics for every dataset and model:** accuracy, macro-F1, per-class precision/recall/F1, ROC-AUC and PR-AUC where appropriate, and calibration.
- **BREAKING: thresholding is simplified and the genetic algorithm is removed.** For binary tasks the rule is "predict positive when P(positive | x) ≥ t". Precision, recall, accuracy and positive-class F1 are plotted against t, with decision cost in a separate panel and three marked thresholds: default, analytic cost threshold, validation-selected.
- **BREAKING: the confidence-triage experiment and its paper section are removed.**
- **BREAKING: perturbation explanations run live on Jev.** KernelSHAP, LIME and the deletion test run on Jev for a seeded sample from each dataset, replacing the mock-only pipeline.
- **Importance sampling is refocused on accuracy estimation.** A fixed pool with cached predictions, labels hidden from the sampler, uniform versus confidence-guided sampling at several budgets, every sampling probability positive, and an inverse-probability estimator. The macro-F1 estimator is dropped.
- **Paper revised and extended.** Five datasets, zero-shot Jev explicitly separated from the supervised baseline, prior public Jev evaluations cited only after checking each number against its source, an appendix with the five exchanges and the dataset documentation, and a check that every reported number matches the saved outputs. A short report states findings, limitations, actual costs and anything left incomplete.
- **Saved artifacts are kept.** Predictions, payload examples, tables and figures are stored in tracked files; the response cache and raw log are backed up outside the working copy.

Prior public evaluations found during planning (each to be re-verified against its source before it is cited):

- Deußer, Sparrenberg and Sifa, "Evaluating and Benchmarking the System One Model Jev", arXiv:2609.37647, on `jev-1.13.0`: SST-2 validation (872), AG News test (7,600) and Banking77 test (3,076).
- The `dhruvmehra/jevbench` repository: SST-2, AG News and Banking77 on 500-item samples.
- German Credit: no public Jev evaluation found.
- Breast Cancer Wisconsin Diagnostic: only a second-hand mention of an exploratory tabular benchmark (in arXiv:2609.34024); the primary source has not been located.

## Capabilities

### New Capabilities
- `supervised-baseline`: the XGBoost baseline, its SageMaker processing and training pipeline, leakage controls, label-budget accounting, the learning curve, and the lifecycle and cost of the AWS resources it needs.

### Modified Capabilities
- `experiment-harness`: five datasets with frozen evaluation IDs and documented provenance; two models (Jev and the baseline); the Jev cap raised to 60,000; raw request/response logging; persisted predictions; mock mode covering all datasets and both models; dry-run reporting scoped to stages that make remote calls.
- `classification-evaluation`: metrics, calibration and uncertainty reported for every dataset and model, including multiclass and repeated-split datasets.
- `cost-sensitive-thresholding`: the threshold rule, the threshold sweep figure and the three marked thresholds; the genetic algorithm requirement is removed.
- `confidence-triage`: all requirements removed; the capability is retired.
- `importance-sampling-evaluation`: accuracy-only estimation on a fixed pool with labels hidden from the sampler.
- `perturbation-explainability`: attributions and faithfulness evaluated live on Jev for every dataset instead of on the mock.
- `paper-manuscript`: new structure, request/response appendix, prior-work citations, zero-shot versus supervised distinction, label budgets, actual costs, and number verification.

## Impact

- **Code:** `src/ciayn/data/` (four new loaders, split freezing), `src/ciayn/backends/` (exchange log, budgets, a baseline prediction reader, mock generalised to text and many classes), `src/ciayn/experiments/` (metrics, thresholding, importance, explain rewritten; triage and the genetic algorithm deleted), new `src/ciayn/baseline/` (SageMaker pipeline and job scripts), `src/ciayn/paper/`, `paper/`, `configs/`, `Makefile`, tests.
- **New infrastructure:** `infra/` CloudFormation template for the S3 bucket, the SageMaker execution role and the permissions added to `agentcore-workshop-role` in `us-east-2`. Estimated AWS cost under USD 5.
- **Dependencies:** `datasets` (Hugging Face) returns; `boto3` is added. `xgboost` produces results only inside SageMaker; locally it is a development-only dependency for testing the training script on toy data.
- **External calls:** up to 60,000 Jev requests (about 57,700 planned); SageMaker processing and training jobs; downloads from Hugging Face and OpenML.
- **Existing results:** the 60-item German Credit results are superseded. The 99 cached responses are reused; nothing already spent is re-queried.
- **Specs:** the archived design's "Jev is the only model", "German Credit only", "100 inferences" and "explainability is mock-only" decisions are reversed by this change.
