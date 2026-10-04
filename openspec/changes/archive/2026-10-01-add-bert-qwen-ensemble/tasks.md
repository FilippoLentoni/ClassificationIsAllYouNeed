# Tasks

## 1. Four-model registry (offline)

- [x] 1.1 Set `config.MODELS` to jev, qwen, xgb, bert with `ZERO_SHOT` and `SUPERVISED`; add labels and colours in `experiments/common.py` and `paper/build.py`; verify with `pytest tests/test_metrics.py tests/test_paper.py` and that `metrics --mode live` on the current results still produces byte-identical Jev and XGBoost outputs with qwen and bert marked missing.
- [x] 1.2 Generalise the learning curve, label budgets and text-baseline sentence from `"xgb"` to every supervised model; verify with a new test in `tests/test_metrics.py` that a mock BERT curve is read and plotted next to XGBoost.
- [x] 1.3 Enforce identical identifiers across any pair of models and add separate scoring of `qwen_think` on its subsample; verify with tests that a mismatched identifier set raises and that the subsample comparison uses only subsample identifiers.
- [x] 1.4 Add `mock_model` backends for qwen and bert (bert with a learning curve) and run the whole mock pipeline; verify `paper --mode mock` builds with four models and the mock watermark, and `pytest` passes.
- [x] 1.5 Add a test that none of the new stages changes the Jev ledger; verify it passes.

## 2. Configuration, prices and AWS template

- [x] 2.1 Add `configs/encoder.yaml` and `configs/llm.yaml` (revisions, images, instances, regions, runtimes, grid, sequence lengths, subsample sizes); set `aws_spend_limit_usd: 75.0`; add the GPU prices with source and date; verify with a config-loading test and `baseline plan --mode live` showing the new limit.
- [x] 2.2 Put all three pipelines on one attempts record with a `model` field (existing records become `xgb`) and share the spend guard; verify with tests that the recorded total includes every earlier execution and that a job whose worst case exceeds the remaining limit is refused.
- [x] 2.3 Extend `infra/sagemaker.yaml` for the `ciayn-bert` and `ciayn-qwen` pipelines; verify with `cfn-lint` or `aws cloudformation validate-template` and a test that the policy names only the three pipelines.
- [x] 2.4 Show the template change to the user and deploy only after approval; verify the stack update completes and record it in `results/live/aws_stack.json`.
- [x] 2.5 Update `docs/AWS_BASELINE.md` with the new jobs, the processing-quota substitution, costs and teardown; verify the teardown command in the doc covers the weights prefix.

## 3. BERT baseline

- [x] 3.1 Write the local packager (serialized text, labels for training and validation, evaluation rows without labels, nested fraction indices shared with XGBoost); verify with a test that no evaluation label is in the uploaded files and that fraction indices equal XGBoost's.
- [x] 3.2 Write `bert_train.py` (grid on validation log-loss, refit per fraction and repeat, bf16, timing, truncation counts) and run it locally on CPU on a tiny slice in a test; verify probabilities sum to 1 and the selected configuration is saved.
- [x] 3.3 Write the `ciayn-bert` pipeline definition; verify with a test that two generations from the same configuration are identical and that the image is the pinned Hugging Face PyTorch training container.
- [x] 3.4 Add `encoder run|fetch` to the CLI with dry run, instance/region iteration, running markers and identifier checks on ingestion; verify with fake-SageMaker tests.
- [x] 3.5 Live smoke: German Credit, repeat 0, one fraction; verify the job succeeds, cost is recorded and the predictions pass the identifier check.
- [x] 3.6 Live run on all five datasets; verify five succeeded executions, saved predictions and learning curves, and the recorded total below the limit.

- [x] 3.7 Rerun XGBoost with inference timing on its recorded combination; verify the new predictions equal the saved ones (or record every difference) and that `infer_s_per_item` is saved for every dataset.

## 4. Qwen baseline

- [x] 4.1 Render Qwen prompts from the frozen Jev templates for every item Jev predicted (no labels), and save one rendered prompt per dataset; verify with a test that each prompt contains the item's Jev state, instructions and all options in Jev's order.
- [x] 4.2 Write the staging script (idempotent download of the pinned revision to S3 with a sha256 list) and the `qwen_infer.py` scorer (option scoring with end-of-turn token, thinking disabled and enabled, timings, per-dataset checkpoints, single-request latency sample, invalid items as uniform); verify the scorer's normalisation and prefix-option handling with a fake-logprob test.
- [x] 4.3 Write the `ciayn-qwen` pipeline definition (processing step on CPU, training step on GPU for inference); verify determinism and that the training step is documented as the inference substitution in the saved run.
- [x] 4.4 Add `llm stage|smoke|run|fetch|project-thinking` to the CLI with dry run, spend guard, instance/region iteration and identifier checks; verify with fake-SageMaker tests, including refusal when the projected cost exceeds the remaining limit.
- [x] 4.9 Add the `escalation-val` stage (500 seeded validation items each for AG News and Banking77, stage budget 1,000, saved apart from the frozen Jev items) and include those items in Qwen's full run; verify with tests that no evaluation item is requested and that the budget is enforced, then run it live and check the ledger grew by at most 1,000.
- [x] 4.5 Live staging of the weights; verify the checksum list matches the pinned revision and a second run downloads nothing.
- [x] 4.6 Live smoke on 20 items per dataset; verify the model loads, whether the prefix cache is reused, tokens per second, and the projected full-run cost; switch to the transformers fallback if needed and record which method is used.
- [x] 4.7 Live full run (thinking disabled on all items, latency sample, thinking-enabled subsample) only if the projection fits the limit; verify saved predictions for all five datasets, identifier checks, and recorded load, inference and billable seconds.
- [x] 4.8 Run `project-thinking`; run thinking enabled on all items only if the projection fits the remaining limit, otherwise record the refusal; verify the decision and its numbers are saved.

## 5. Comparison and escalation

- [x] 5.1 Implement the `comparison` stage (F1 by training labels figure and table; cost per 1,000 predictions, training cost, latency by kind); verify with tests that every cost traces to saved token counts or billable seconds and that Qwen cost is given with and without load time.
- [x] 5.2 Implement the `escalation` stage (curve over distinct Jev confidences, endpoints, validation-selected threshold per repeat, unreachable-error count, cost and latency per share); verify with tests for the endpoints, label-permutation invariance of the threshold, and no remote calls.
- [x] 5.3 Run `metrics`, `thresholding`, `comparison` and `escalation` in live mode one at a time; verify outputs exist for every dataset and model and that Jev and XGBoost numbers are unchanged from the current paper.

## 6. Paper and report

- [x] 6.1 Generate the four-model tables, the thinking-subsample table, the comparison and escalation figures and tables, the Qwen prompt appendix and the new macros; verify `verify-paper --mode live` passes.
- [x] 6.2 Write the two new sections and update abstract, introduction, setup (models, substitution, costs), limitations and conclusion from the saved outputs only; verify every number in them is a generated macro.
- [x] 6.3 Build the PDF, inspect every page and update `docs/PDF_INSPECTION.md`; verify no overfull boxes or unresolved references.
- [x] 6.4 Regenerate `docs/REPORT.md` (findings, actual AWS cost by model, the thinking decision, anything incomplete) and update `README.md` commands; verify `scripts/fresh_checkout_check.sh` passes.
- [x] 6.5 Back up the new results and run records; verify the backup record lists them.
- [ ] 6.6 When the user confirms, tear down the AWS resources including the staged weights; verify the stack and bucket are gone.
