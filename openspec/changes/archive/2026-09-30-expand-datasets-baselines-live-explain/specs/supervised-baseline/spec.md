# Spec Delta

## Purpose

Provides a supervised XGBoost baseline for every dataset, trained and scored through an Amazon SageMaker pipeline, so that zero-shot Jev can be compared with a conventional classifier on identical examples without test-label leakage.

## ADDED Requirements

### Requirement: Identical evaluation examples
The baseline SHALL produce a class-probability vector for exactly the frozen evaluation examples of each dataset, and for every repeat of the repeated-split datasets, keyed by the same example identifiers used for Jev.

#### Scenario: Same identifiers as Jev
- **WHEN** baseline predictions are read for a dataset
- **THEN** their identifiers equal the frozen evaluation identifiers, and reading fails if any is missing or extra

### Requirement: Preprocessing fitted on training data only
All preprocessing that learns from data (category encodings, vocabularies, feature weights, scaling) SHALL be fitted on the training part only and then applied unchanged to validation and evaluation examples.

#### Scenario: No evaluation data in fitting
- **WHEN** preprocessing is fitted for a dataset or a split repeat
- **THEN** only training examples are used, and a category or word that appears only in validation or evaluation examples has no fitted entry

### Requirement: Tuning without test labels
Model selection, early stopping and any hyperparameter choice SHALL use training and validation labels only. Evaluation labels SHALL NOT be provided to the training pipeline.

#### Scenario: Evaluation labels withheld
- **WHEN** data is prepared for the training pipeline
- **THEN** the evaluation examples are supplied without their labels, and metrics are computed afterwards from the returned probabilities

### Requirement: SageMaker pipeline orchestration
Baseline preprocessing and training SHALL run as a SageMaker pipeline with a processing job and a training job for each dataset. The pipeline definition, the job parameters, the container versions and the execution identifiers SHALL be recorded. When the preferred instance type or region cannot run the jobs, the stage SHALL try the configured alternative instance types and then the configured alternative regions, in order, without asking, recording every attempt and its outcome. Only when every configured combination has failed SHALL the stage stop and report, and no baseline result SHALL be produced by a route other than SageMaker without that substitution being recorded and stated.

#### Scenario: Pipeline run recorded
- **WHEN** a baseline pipeline execution completes for a dataset
- **THEN** its execution identifier, job names, instance types, container versions, hyperparameters and billable seconds are saved with the predictions

#### Scenario: Instance type or region not available
- **WHEN** the account refuses to start a job on an instance type or in a region (for example zero quota or no capacity)
- **THEN** the stage moves to the next configured instance type, and after the instance types are exhausted to the next configured region, and records the refused attempt with its error

#### Scenario: All datasets on one combination
- **WHEN** a working instance type and region are found
- **THEN** all five datasets are run on that combination, and the instance type and region used are saved with the predictions and stated in the paper

#### Scenario: SageMaker unavailable
- **WHEN** every configured instance type in every configured region has refused the jobs
- **THEN** the stage stops, reports every attempt and its error, and produces no baseline predictions

#### Scenario: Reproducible definition
- **WHEN** the pipeline definition is generated twice from the same configuration
- **THEN** the two definitions are identical

### Requirement: Label budget accounting
The baseline SHALL record, for each dataset and split repeat, the number of labels used for training, for early stopping and hyperparameter selection, and for threshold selection.

#### Scenario: Budgets saved
- **WHEN** a baseline run completes
- **THEN** the saved results contain the training, validation and threshold-selection label counts

### Requirement: Learning curve
The baseline SHALL be trained on increasing fractions of the training data with the selected hyperparameters and scored on the evaluation examples, producing a learning curve for each dataset. If the curve cannot be produced for a dataset, that SHALL be recorded with the reason.

#### Scenario: Curve produced
- **WHEN** the learning-curve run completes for a dataset
- **THEN** a figure and a table give evaluation accuracy and macro-F1 at each training-set size, with Jev's zero-shot result as a reference line

#### Scenario: Curve not feasible
- **WHEN** the learning curve cannot be produced for a dataset
- **THEN** the results record that it was skipped and why, and the paper does not show a curve for that dataset

### Requirement: AWS resource lifecycle and cost
The AWS resources the baseline needs SHALL be defined in one reviewable template that can be deployed in any of the configured regions, SHALL be created only after the user has approved that template, SHALL grant no more access than the baseline needs, and SHALL be removable with one documented command that covers every region in which they were created. A spending limit SHALL be configured and checked before any job is started, and the actual cost of the jobs SHALL be recorded.

#### Scenario: Approval before creation
- **WHEN** the resources have not been approved by the user
- **THEN** nothing is created in the AWS account

#### Scenario: Spending limit
- **WHEN** the estimated cost of the jobs about to start, added to the cost already recorded, exceeds the configured limit
- **THEN** no job is started and the estimate and the limit are reported

#### Scenario: Teardown
- **WHEN** the documented teardown command is run
- **THEN** the storage bucket, the execution role and the permissions added for the baseline are removed in every region where they were created
