# Design

## Context

See proposal.md for the motivation. Current state that shapes the approach:

- The model registry is `config.MODELS = ("jev", "xgb")`. `metrics`, `thresholding`, `common` and
  `paper/build.py` loop over it but hard-code `"xgb"` for the learning curve, the label budgets and
  the text-baseline sentence.
- The XGBoost baseline is one SageMaker pipeline (`ciayn-xgb`) defined as JSON through boto3,
  packaged locally (`baseline/package.py`, evaluation rows uploaded without labels), with a spend
  guard (`check_spend`), instance/region iteration (`find_combination`), attempt records
  (`results/live/baseline_attempts.json`) and ingestion that checks identifiers (`check_ids`).
- The CloudFormation stack `ciayn-baseline` (us-east-2) gives the execution role S3 access to one
  bucket, ECR pull from any registry, and job permissions for `pipelines-*` jobs; the operator
  policy allows creating and running pipelines. Changes to it need the user's approval.
- Account facts, checked on 2026-09-30: every GPU **processing**-job quota is 0 in us-east-2,
  us-east-1 and us-west-2; GPU **training**-job quotas are 1 for ml.g5.xlarge, ml.g5.2xlarge,
  ml.g5.8xlarge, ml.g5.12xlarge, ml.g6e.xlarge and ml.g6e.12xlarge in us-east-2. The pricing API is
  denied to the machine's role; AWS's public price file gives, per hour in us-east-2: ml.g5.xlarge
  1.2575, ml.g5.2xlarge 1.515, ml.g5.12xlarge 7.09, ml.g6e.xlarge 2.61, ml.g6e.12xlarge 13.12.
- Containers present in us-east-2: `huggingface-pytorch-training:2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04`
  and `vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.2`.
- `Qwen/Qwen3.6-27B` (revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, Apache 2.0, not gated) is
  a 27.8B-parameter hybrid model (full and linear attention layers) published as image-text-to-text;
  about 56 GB in bf16. `google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
- Jev's items: every row of the tabular datasets, and evaluation plus `jev_val` items for text
  (13,620 in total, 11.8M input tokens).
- The machine has 3.8 GB of memory; heavy local stages must run one at a time.

## Goals / Non-Goals

**Goals:**
- Four models scored by the same code on the same identifiers, with the zero-shot/supervised
  distinction kept everywhere.
- Every cost and latency figure traceable to recorded token counts, billable seconds or timings.
- Qwen's per-item distribution computed by one documented scoring method, with the same method in
  both thinking modes.

**Non-Goals:**
- No new Jev requests; no re-tuning of Jev templates.
- No recalibration of any model, no BERT large or other encoders, no Qwen quantization.
- No hosted Qwen API and no real-time endpoint; latency is measured inside the batch job.
- Importance sampling and perturbation explanations stay Jev-only.

## Decisions

### D1. Model registry and result layout
`config.MODELS = ("jev", "qwen", "xgb", "bert")`, with `ZERO_SHOT = ("jev", "qwen")` and
`SUPERVISED = ("xgb", "bert")`. Results go to `results/live/<ds>/<model>/predictions.parquet` as today.
Thinking-enabled Qwen is saved as model id `qwen_think`, which is **not** in `MODELS`: it appears
only in the thinking-subsample table and never in a four-model table. All `"xgb"`-specific code
(learning curve, label budgets, text-baseline sentence) is generalised to "every supervised model".
*Alternative:* one model with a mode column; rejected because it would let the two modes mix.

### D2. BERT on SageMaker
Pipeline `ciayn-bert`, one **training** step per dataset (the user asked for a training job; data
is packaged locally exactly as for XGBoost, so no processing step is needed). Inputs: serialized
text and labels for training and validation parts; evaluation rows without labels. Tabular rows use
Jev's serialization (`data/serialize.py`), so BERT reads what Jev reads.
- Model: bert-base-uncased at the pinned revision, `AutoModelForSequenceClassification`, bf16,
  batch 32, AdamW, linear warm-up 10%, seed 0.
- Selection on validation only: learning rate in {2e-5, 5e-5}, up to 4 epochs, best epoch by
  validation log-loss (the same criterion as XGBoost). The selected configuration is refitted for
  each training fraction in `baseline.yaml` `fractions` (nested, the same indices as XGBoost), and
  for each of the 10 repeats on the tabular datasets (the grid runs on repeat 0 only; its choice is
  reused for repeats 1–9 and recorded as such).
- Maximum sequence length 128 for SST-2 and Banking77, 256 for AG News, 512 for the tabular rows;
  the share of truncated inputs is recorded per dataset.
- Instances, in order: ml.g5.xlarge, ml.g5.2xlarge, ml.g6e.xlarge; regions as for XGBoost.
  Maximum runtime 3 h per job; a fraction that would exceed 80% of it is skipped and recorded.
- Output: evaluation probabilities per repeat and fraction, validation scores, timings for
  training and for evaluation inference (latency per item), truncation counts.
*Alternative:* one job for all datasets; rejected because a failure would lose all five and a
per-dataset job matches the XGBoost records.

### D3. Qwen pipeline: CPU staging, GPU inference in a training step
Pipeline `ciayn-qwen` with two steps:
1. **Processing (CPU, ml.m5.xlarge, 150 GB volume):** if `s3://<bucket>/weights/qwen3.6-27b/<revision>/`
   has a checksum list matching the pinned revision, do nothing; otherwise download the revision
   from Hugging Face (`huggingface_hub.snapshot_download`, no token) and upload it with its sha256
   list. Processing jobs have internet access because they are not attached to a VPC.
2. **Training step used for inference (GPU):** container vLLM 0.29.0; entrypoint `qwen_infer.py`;
   weights channel from S3 (File mode); prompts channel (JSONL, no labels). Order inside the job:
   load model (timed) → thinking disabled on every item of every dataset → single-request latency
   sample → thinking-enabled subsample. Results for each dataset are written to the SageMaker
   checkpoint directory as soon as they are complete, so a stopped job keeps finished datasets.
   One job for all datasets loads the 56 GB model once.
- Instances, in order: ml.g6e.12xlarge (4×L40S, 192 GB), ml.g5.12xlarge (4×A10G, 96 GB); tensor
  parallel 4, bf16, text-only (image inputs disabled), maximum model length 16,384 tokens.
- Maximum runtime 4 h. Worst case on ml.g6e.12xlarge: USD 52.5, which the spend guard checks
  against the recorded total before starting.

### D4. Qwen prompt and scoring method
Prompts are rendered locally from the frozen Jev templates: a system message with Jev's
instructions and the options as `name: description` lines in Jev's order, "Answer with exactly one
option name."; a user message with the Jev request state; Qwen's own chat template with
`enable_thinking=False` (thinking disabled) or `True` (thinking enabled). One rendered prompt per
dataset is saved for the appendix.

The distribution is computed by **option scoring**: for each option, the log-probability of the
continuation `<option name><|im_end|>` after the assistant turn is summed over its tokens, and a
softmax over the options gives the class distribution. The end-of-turn token makes an option that
is a prefix of another a complete answer. With thinking enabled, the model first generates its
reasoning greedily (temperature 0, at most 4,096 thinking tokens; truncation recorded), and the
options are scored after the closing `</think>`.

Scoring needs the prompt's state to be reused across options. vLLM's prefix cache does this; on
hybrid linear-attention models its support is version-dependent. **Smoke run first:** a job on 20
items per dataset measures tokens per second, whether the prefix cache is hit, and the projected
cost of the full run. If the cache is not reused, the fallback is the same scoring in Hugging Face
transformers on the same instance: one prefill per item, then all option suffixes batched on a
copy of the cached state. The method actually used is recorded and stated in the paper.
**Finding of the first smoke run (2026-10-01) and the method now used.** vLLM 0.29 loads the model
and runs on ml.g5.12xlarge (ml.g6e.12xlarge had no capacity in us-east-2). Scoring from prompt
log-probabilities, as first written, skips the prefix cache by design (`skip_reading_prefix_cache`
is set whenever prompt log-probabilities are requested) and ran at about 1,300 tokens per second:
83 s per Banking77 item, a projected 78 hours and USD 554 for the full run, which the spend guard
refused. The scores are now obtained from single-token generation requests, one per edge of the
trie of option tokens, each restricted to the edge's token and returning its raw log-probability
(vLLM's default `raw_logprobs`, the value before the restriction). These requests use the prefix
cache, which vLLM supports for this hybrid architecture (`mamba_cache_mode: align`). The sums are
the same quantity as before; a cross-check on two items per dataset against the prompt
log-probability method is saved with every run. Thinking-enabled items now reason concurrently
(128 per call instead of 16).

**Second finding and the final method (2026-10-01).** For this hybrid architecture vLLM's prefix
cache works in blocks of 784 tokens, so the per-edge requests reuse nothing for prompts shorter
than that and half of a Banking77 prompt; on 4 x 24 GB GPUs only about 12 requests run
concurrently, at about 1,700 prompt tokens per second. Scoring therefore moved to the
transformers engine, which is exact and architecture-independent: the instructions and option
list are computed once per dataset, the rest of the prompt once per item (items of equal length
are batched; padding would change the recurrent state of the linear-attention layers), and all
options on copies of that state. The engine checks on one item per dataset that this equals a
full recomputation. vLLM is used only to generate the reasoning of the thinking subsample, which
the transformers job then scores. Thinking was reduced to 50 items per dataset and 2,048 tokens.
Banking77 is not scored: its 77 options take 4.25 s per item on these GPUs (the user's decision).

*Alternatives:* free generation plus parsing (gives no distribution and can produce invalid
answers, which is what the paper argues against); first-token logprobs over renamed single-token
options (breaks prompt parity with Jev).

### D5. Thinking mode
Thinking disabled on all 13,620 items. Thinking enabled on a seeded subsample of 100 evaluation
items per dataset (500 in total). After the job, `llm project-thinking` computes the cost of
thinking on every item from the subsample's measured seconds per item; `llm run --thinking all` is
refused unless that projection plus the recorded total fits the limit.

### D6. Spend limit, prices and approval
`configs/budgets.yaml` `aws_spend_limit_usd: 75.0`. The GPU prices above are added to the
configured price table with their source and date. The spend guard, worst-case check and attempt
records are shared by all three pipelines (one attempts file with a `model` field; existing XGBoost
records get `model: xgb`). The CloudFormation template gains: the two pipeline names in the
operator policy, `weights/` and `checkpoints/` prefixes (already covered by the bucket policy), and
no new resource types. The updated template is shown to the user and deployed only after approval.
Teardown already empties and deletes the bucket, which removes the staged weights.

### D7. Latency and cost accounting
- Jev: per-request latency and token counts from the saved predictions; cost = tokens × USD 0.042/M.
- Qwen: `load_s`, `infer_s` per dataset and `billable_s`; batch throughput = items / `infer_s`;
  single-request latency from 20 evaluation items per dataset sent one at a time after the batch
  run. Cost per 1,000 predictions is given twice: over the whole billed job, with load time
  allocated to datasets by item count, and over inference time only.
- BERT and XGBoost: training cost from billable seconds; inference latency per item from the timed
  evaluation pass inside the job.

### D8. Comparison and escalation stages
- `comparison`: reads learning curves of the supervised models and headline metrics of the
  zero-shot ones; writes the F1-by-labels figure and table and the cost/latency table.
- **Escalation validation items (added 2026-10-01).** AG News and Banking77 have no Jev predictions
  on validation items. A new live stage `escalation-val` asks Jev for 500 items drawn with a fixed
  seed from each dataset's validation part (stage budget 1,000, same cache, ledger and request log
  as `predict`), saved apart from the frozen `jev_ids` in `results/live/<ds>/jev/escalation_val.parquet`.
  Qwen's full run scores the same items. The frozen split files are not changed.
- **XGBoost inference timing (added 2026-10-01).** `train.py` times batch inference on the
  evaluation items; XGBoost is rerun on its recorded combination and the new predictions are
  compared with the saved ones before they replace them.
- `escalation`: for each dataset and repeat, candidate thresholds are the distinct Jev maximum
  probabilities plus 0 and 1.01; the threshold maximising validation macro-F1 is selected (ties to
  fewer escalations); the evaluation curve, the selected point, and the count of Jev errors at the
  maximum reported confidence are saved. Qwen's validation predictions come from its run on
  `jev_val` (text) and all rows (tabular), which D3 already covers.

### D9. Mock mode and tests
`mock_model` gains `qwen` and `bert`; BERT's mock also writes a learning curve. Fake SageMaker
clients (existing test pattern) cover: pipeline definitions are deterministic; evaluation labels are
absent from uploaded data; spend guard refuses; instance/region iteration; identifier checks on
ingestion; option scoring against a tiny fake scorer (softmax over summed logprobs, prefix
options); escalation endpoints and label-permutation invariance; no Jev ledger change.

### D10. Paper
New sections "Comparison by training labels, cost and latency" and "Escalating low-confidence Jev
predictions"; four-model metrics, calibration and label-budget tables; a thinking-subsample table;
appendix with one Qwen prompt and distribution per dataset; setup states that Qwen inference used
training jobs and why; all numbers generated as macros and checked by `verify-paper`.

## Risks / Trade-offs

- [vLLM 0.29 may not load the Qwen3.6 architecture or reuse prefixes on it] → smoke run before the
  full job; transformers fallback on the same instance; the method used is recorded.
- [Scoring 77 Banking77 options per item is the largest cost] → prefix reuse makes it ~77 short
  suffixes per item; the smoke run projects the cost and the full run is not started if the
  projection plus the recorded total exceeds USD 75.
- [ml.g6e.12xlarge capacity may be unavailable] → fall back to ml.g5.12xlarge (bf16 fits in 96 GB
  with a shorter maximum length), then other regions; quotas outside us-east-2 are not verified.
- [Greedy decoding with thinking differs from Qwen's recommended sampling] → stated as a deviation;
  it makes the thinking results reproducible.
- [BERT on serialized tabular rows is an unusual use of BERT] → reported as such; XGBoost remains
  the tabular reference.
- [Hugging Face download fails or is slow] → the processing step is idempotent and can be rerun;
  weights are fetched once.
- [Jev confidence is rounded to two decimals] → escalation has few distinct thresholds; the
  unreachable errors are counted and reported (spec requirement).
- [Local memory 3.8 GB] → packaging and metrics run per dataset, one stage at a time.

## Migration Plan

1. Generalise the model registry and stages; existing live results are unchanged and re-verified.
2. Update the template, show it, deploy after approval.
3. BERT smoke (German Credit, one repeat, one fraction), then all datasets.
4. Qwen staging, Qwen smoke, projection check, full Qwen run.
5. Comparison, escalation, metrics, paper, verification.
Rollback: the new models are additive; removing `qwen` and `bert` from `MODELS` restores the
current paper. Teardown removes all AWS resources including the weights.

## Open Questions

- Exact per-item throughput and therefore the final Qwen cost; measured by the smoke run and
  checked against the limit before the full run.
