# Proposal

## Why

The benchmark compares zero-shot Jev only with a TF-IDF/one-hot XGBoost baseline, which is a weak
supervised reference on text and says nothing about open-weight language models. The user wants two
more reference points, fine-tuned BERT and self-hosted Qwen3.6-27B, compared with Jev and XGBoost
on F1, latency and cost as a function of the number of training labels, and wants to know what
escalating Jev's low-confidence predictions to Qwen would do to F1.

## What Changes

- Add a **fine-tuned BERT baseline** (`google-bert/bert-base-uncased` with a classification head,
  pinned revision) on all five datasets, trained as a SageMaker pipeline training job on GPU, with
  model selection on validation labels only, and a learning curve over the same training fractions
  as XGBoost. Tabular rows are given to BERT as the same serialized text Jev receives.
- Add a **self-hosted Qwen3.6-27B zero-shot model** (`Qwen/Qwen3.6-27B`, Apache 2.0, pinned revision)
  on all five datasets, served with vLLM in SageMaker. A CPU processing step stages the weights from
  Hugging Face to S3 and writes the prompts; the GPU inference runs as a **training-job step**,
  because every GPU processing-job quota in the account is 0 (us-east-2, us-east-1, us-west-2) while
  GPU training-job quotas exist. This departs from the "processing job only" wording of the
  request and is stated in the paper. Qwen sees exactly Jev's frozen instructions, option
  descriptions and serialized inputs, and returns a probability for every class.
- **Qwen runs on four datasets, not on Banking77 (decided 2026-10-01).** Measured on the GPUs that
  had capacity (4 x 24 GB), scoring Banking77's 77 options takes 4.25 s per item, about 6 GPU
  hours and USD 35 for that dataset alone. The user chose not to run it. The paper marks Qwen on
  Banking77 as not available and says why; the escalation ensemble is reported for the other four.
- **Thinking mode.** Qwen runs with thinking off on every item. Thinking on runs on a fixed seeded
  subsample; its measured GPU time is projected to all items and thinking on runs on every item
  only if that projection fits the AWS spending limit. Self-hosting has no per-token fee, but
  thinking lengthens GPU time, which is what is billed.
- Add a **cost and latency comparison** of Jev, Qwen, BERT and XGBoost: F1 against training labels
  (BERT and XGBoost curves, Jev and Qwen as zero-shot lines), cost per 1,000 predictions, training
  cost, and latency, with batch throughput and single-request latency reported separately.
- Add a **Jev→Qwen escalation ensemble**, computed offline from saved predictions: items whose Jev
  confidence is below a threshold take Qwen's prediction. Report macro-F1 against the share of
  items escalated, the threshold selected on validation items only, and its cost and latency.
- Extend metrics, calibration, thresholding and the paper tables to four models; label Jev and Qwen
  zero-shot and XGBoost and BERT supervised.
- Raise the AWS spending limit from USD 5 to USD 75 for the whole study (USD 0.42 already spent);
  add GPU instance prices and fallbacks to the configuration.
- **1,000 new Jev requests, nothing else.** AG News and Banking77 have no Jev predictions on
  validation items, so the escalation threshold could not be selected there without evaluation
  labels. On 2026-10-01 the user chose to add Jev predictions for 500 seeded items from each of
  their existing validation parts (ledger 49,000 -> at most 50,000 of the 60,000 cap). Every other
  Jev result comes from the saved predictions.
- **XGBoost rerun with inference timing.** The XGBoost jobs never timed inference; on 2026-10-01
  the user chose to rerun them (about USD 0.45) so that XGBoost latency is measured. Their
  predictions are expected to be identical and are checked against the current ones.
- **BREAKING (spec):** the harness no longer restricts the study to exactly two models, and the
  paper gains an escalation section although it previously excluded a triage section.

## Capabilities

### New Capabilities
- `fine-tuned-encoder-baseline`: BERT fine-tuned on SageMaker for every dataset, with validation-only selection, label budgets and a learning curve.
- `open-weight-llm-baseline`: self-hosted Qwen3.6-27B, zero-shot, on SageMaker; weights staging, prompt parity with Jev, full class distribution, thinking-mode handling, recorded GPU time and cost.
- `model-cost-latency-comparison`: F1 by training labels, cost per prediction and latency for all four models, measured from saved outputs.
- `ensemble-escalation`: offline Jev→Qwen escalation with a validation-selected confidence threshold and an F1-versus-escalation-share curve.

### Modified Capabilities
- `experiment-harness`: the prediction interface covers four models instead of exactly two.
- `classification-evaluation`: metrics and calibration are computed for every model on identical examples.
- `supervised-baseline`: the spending limit and teardown cover all SageMaker work of the study, including the new GPU jobs and staged weights.
- `paper-manuscript`: structure adds the model comparison and the escalation section; provenance covers Qwen and BERT versions, GPU time and cost; zero-shot and supervised labelling covers four models.

## Impact

- **Code:** `src/ciayn/config.py` (model registry), new `src/ciayn/encoder/` and `src/ciayn/llm/` packages
  (pipeline definitions and job scripts), new `src/ciayn/experiments/escalation.py` and
  `comparison.py`, generalised `metrics.py`, `thresholding.py`, `common.py` and `paper/build.py`.
- **Configuration:** new `configs/encoder.yaml`, `configs/llm.yaml`; `configs/budgets.yaml` (AWS limit
  75); price table for GPU instances.
- **AWS:** two more SageMaker pipelines in the existing stack's bucket and role; about 56 GB of Qwen
  weights in S3 until teardown; CloudFormation template changes need the user's approval.
  Containers: AWS Hugging Face PyTorch training DLC (transformers 5.3.0) and AWS vLLM DLC 0.29.0.
- **Estimated AWS cost:** USD 25-60 with thinking off, within a USD 75 limit.
- **Paper:** new tables, figures and two sections; appendix gains one Qwen prompt and response per dataset.
- **Jev:** none.
