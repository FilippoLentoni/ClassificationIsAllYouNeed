# open-weight-llm-baseline Specification

## Purpose

Provides a self-hosted open-weight language model, Qwen3.6-27B, queried zero-shot on every dataset with the same instructions Jev receives, so that Jev can be compared with a frontier open model on accuracy, calibration, latency and cost.

## Requirements

### Requirement: Pinned open weights, self-hosted
The study SHALL run a pinned revision of `Qwen/Qwen3.6-27B` on SageMaker GPU instances in the project's account, from weights copied once from Hugging Face into the project's S3 bucket. No hosted third-party inference API SHALL be used. The model identifier, revision, weight checksum list, serving engine version and container image SHALL be saved with the predictions.

#### Scenario: Weights staged once
- **WHEN** the weights for the pinned revision are already in the bucket with a matching checksum list
- **THEN** they are not downloaded again

#### Scenario: Provenance saved
- **WHEN** Qwen predictions are saved for a dataset
- **THEN** the model revision, serving engine version, container image, instance type and region are saved with them

### Requirement: Prompt parity with Jev
For every item, Qwen SHALL receive the same serialized input, instructions, option names, option descriptions and option order as the frozen Jev template for that dataset, wrapped in the model's chat format. The exact rendered prompt for one evaluation example per dataset SHALL be saved.

#### Scenario: Same content as Jev
- **WHEN** a Qwen prompt is rendered for an item
- **THEN** it contains the item's Jev request state, the dataset's Jev instructions and every option with its description in Jev's order

### Requirement: Full class distribution from Qwen
For every item Qwen SHALL provide a probability for every class of the dataset, obtained by scoring each option's label as the model's answer and normalising over the options, and the argmax label. An item whose scoring fails SHALL be recorded as an invalid output with a uniform distribution, not mapped to a class.

#### Scenario: Distribution over all classes
- **WHEN** Qwen predictions are read for a dataset
- **THEN** each row has a probability for every class in the dataset's class order summing to 1 within 1e-6

#### Scenario: Failed item
- **WHEN** scoring fails for an item
- **THEN** it is recorded as invalid with a uniform distribution and counted in the invalid-output total

### Requirement: Qwen on every item Jev predicted
Qwen SHALL be run on exactly the items Jev predicted for each dataset it is run on: the evaluation identifiers and the validation items used for Jev's threshold or escalation-threshold selection. A dataset on which Qwen is not run SHALL be listed with the measured reason, and SHALL be reported as not available, never filled in.

#### Scenario: Identifier match
- **WHEN** Qwen predictions are read for a dataset
- **THEN** their identifiers equal the identifiers of the saved Jev predictions, and reading fails otherwise

#### Scenario: Dataset not run
- **WHEN** a dataset is configured as not scored by Qwen
- **THEN** no Qwen job covers it, the saved results and the paper show Qwen as not available for it, and the reason is stated

### Requirement: Thinking mode handling
Qwen SHALL run with thinking disabled on every item. It SHALL also run with thinking enabled on a fixed, seeded subsample of each dataset's evaluation items. Thinking enabled SHALL be run on all items only if the GPU cost projected from the subsample's measured time fits within the remaining AWS spending limit. The two modes SHALL be saved and reported separately and never mixed in one result.

#### Scenario: Subsample first
- **WHEN** thinking-enabled inference is requested for all items
- **THEN** the projected cost from the subsample is compared with the remaining limit first, and the run is refused with the projection and the limit reported if it does not fit

#### Scenario: Modes kept apart
- **WHEN** Qwen results are reported
- **THEN** each figure and table states whether thinking was enabled, and thinking-enabled results cover exactly the items they were run on

### Requirement: Qwen jobs on SageMaker GPU
Weight staging and prompt preparation SHALL run as a SageMaker processing job on CPU. Inference SHALL run as a SageMaker training-job step on GPU, because the account has no GPU quota for processing jobs; this substitution SHALL be recorded and stated in the paper. The stage SHALL try the configured instance types and then the configured regions in order without asking, record every attempt, and record billable seconds, model-loading time and inference time separately.

#### Scenario: Substitution stated
- **WHEN** Qwen inference runs in a training-job step
- **THEN** the saved run and the paper both state that a training job was used for inference and why

#### Scenario: Time recorded
- **WHEN** a Qwen execution completes
- **THEN** its billable seconds, model-loading seconds and inference seconds are saved, together with the number of items and tokens processed
