# SageMaker jobs: XGBoost, BERT and Qwen3.6-27B

Three SageMaker pipelines run every AWS job of the study, in one CloudFormation stack per region.
Nothing is created or changed in AWS until the template, or a change to it, has been approved.

| Pipeline | Steps | Container | Instance (first choice) |
|---|---|---|---|
| `ciayn-xgb` | processing job (features), training job (XGBoost) per dataset | SageMaker scikit-learn 1.4-2, XGBoost 3.0-5 | ml.m5.xlarge |
| `ciayn-bert` | training job per dataset (fine-tune BERT) | AWS Hugging Face PyTorch training 2.9.0, transformers 5.3.0 | ml.g5.xlarge |
| `ciayn-qwen` | processing job (copy weights to S3, once), then a **training job used for inference** | SageMaker scikit-learn; AWS vLLM 0.29.0 | ml.m5.xlarge; ml.g6e.12xlarge |

**Why Qwen's inference is a training job.** On 2026-09-30 every GPU quota for SageMaker
*processing* jobs in this account was 0 in us-east-2, us-east-1 and us-west-2, while GPU
*training*-job quotas were 1 for ml.g5.xlarge, ml.g5.2xlarge, ml.g5.12xlarge, ml.g6e.xlarge and
ml.g6e.12xlarge. The inference step therefore runs as a training job in the same pipeline. The
substitution is saved with every Qwen run and stated in the paper.

## What the template creates

[`infra/sagemaker.yaml`](../infra/sagemaker.yaml) is one CloudFormation stack, `ciayn-baseline`,
per region used:

| Resource | Name | What it allows |
|---|---|---|
| S3 bucket | `ciayn-baseline-<account>-<region>` | Private, encrypted, public access blocked. Holds job inputs and outputs (deleted after 30 days), the staged Qwen weights (about 56 GB, kept until teardown), Qwen's checkpoints, and the backup of the Jev cache, ledger and request log. |
| IAM role | `ciayn-sagemaker-exec-<region>` | Assumed by SageMaker only. Reads and writes that bucket, writes its own logs and metrics, pulls container images, and starts the pipelines' own jobs. |
| IAM policy | `ciayn-sagemaker-operator-<region>`, attached to `agentcore-workshop-role` | Lets this machine create and run exactly the pipelines `ciayn-xgb`, `ciayn-bert` and `ciayn-qwen`, read and stop their jobs and logs, pass only the execution role, and read SageMaker quotas. |

Every run checks that the deployed stack was built from the current `infra/sagemaker.yaml` (by its
sha256) and refuses to start otherwise. A changed template is applied with
`baseline update --yes`, only after the user has approved the change.

## Commands

```bash
uv run python -m ciayn baseline plan --mode live            # what exists and what it costs; starts nothing
uv run python -m ciayn baseline deploy --mode live --yes    # first deployment, after approval
uv run python -m ciayn baseline update --mode live --yes    # apply an approved template change
uv run python -m ciayn baseline run --mode live             # XGBoost: one execution per dataset
uv run python -m ciayn baseline fetch --mode live

uv run python -m ciayn encoder plan --mode live             # BERT: worst case per dataset
uv run python -m ciayn encoder run --mode live              # one execution per dataset, instance/region iteration
uv run python -m ciayn encoder fetch --mode live

uv run python -m ciayn llm plan --mode live                 # Qwen: worst cases and the saved projection
uv run python -m ciayn llm stage --mode live                # copy the pinned weights to S3 (skipped if present)
uv run python -m ciayn llm smoke --mode live                # 20 items per dataset; saves the cost projection
uv run python -m ciayn llm run --mode live                  # full run, only if the projection fits the limit
uv run python -m ciayn llm fetch --mode live
uv run python -m ciayn llm project-thinking --mode live     # thinking enabled on every item: does it fit?
uv run python -m ciayn llm run --mode live --thinking all   # only if that projection fits

uv run python -m ciayn baseline teardown --mode live --yes  # everything, including the weights
```

Every `run`, `smoke` and `stage` accepts `--dry-run`, which reports the jobs it would start and
starts nothing.

## Which instance type and region

`configs/baseline.yaml`, `configs/encoder.yaml` and `configs/llm.yaml` list the instance types
and regions in the order they are tried. The first execution of each model probes them: if an
instance type or region is refused (zero quota, no capacity), the next one is tried without
asking and the attempt is recorded in `results/live/baseline_attempts.json`, which holds the
attempts of all three models. A failure in our own job code does not trigger the iteration; it
stops the stage. The combination that worked is saved (`baseline_combination.json`,
`encoder_combination.json`, `llm_combination.json`) and stated in the paper.

## Settings

- **XGBoost.** Grid of tree depth {3, 6} and learning rate {0.1, 0.3}, early stopping on the
  validation part (at most 500 rounds, patience 30), lowest validation log-loss kept. Text
  datasets use 16 histogram bins. Batch inference time per evaluation item is recorded.
- **BERT.** `google-bert/bert-base-uncased` at a pinned revision, learning rate {2e-5, 5e-5}, up
  to 4 epochs, best (learning rate, epoch) by validation log-loss on repeat 0 and reused for the
  other repeats; nested training fractions identical to XGBoost's. Inputs are the texts Jev
  receives, truncated at 512 (tabular), 256 (AG News) or 128 tokens; the truncated share is
  recorded.
- **Qwen.** `Qwen/Qwen3.6-27B` at a pinned revision, bf16, tensor parallel 4. Every option is
  scored as the answer and the scores are normalised over the options. Thinking disabled on every
  item; thinking enabled on 100 seeded evaluation items per dataset; 20 items per dataset sent one
  at a time for single-request latency.

## What is uploaded

Only public dataset rows. XGBoost and BERT receive training and validation rows with labels and
evaluation rows without them; Qwen receives prompts without any label. For SST-2 and AG News,
evaluation labels never leave this machine. For the two repeated-split datasets a row
that is a test row in one repeat is a training row in another, so its label is uploaded for that
other repeat; each repeat is fitted only from its own files. All metrics are computed locally.

## Cost and the spending limit

`aws_spend_limit_usd` in `configs/budgets.yaml` is 75 (raised from 5 on 2026-10-01 at the
user's choice). It covers every SageMaker job of the study. Before each execution, the cost
already recorded for all three models plus the worst case of the next execution (every job
running to its stopping condition, at list price) must stay under the limit, or nothing is
started. Qwen's full run also needs the projection from its smoke run to fit. Actual cost is
computed from billable seconds and list prices (`configs/baseline.yaml`, read from AWS's public
price file on 2026-09-30) and saved with each run.

## Teardown

`baseline teardown --yes` deletes the pipelines, empties the bucket (including the staged Qwen
weights, checkpoints and the backup) and deletes the stack in every region where one exists. It
refuses to run unless the local cache, ledger and request log are identical to the latest backup,
because the backup in the bucket is deleted with it.
