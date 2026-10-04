## Purpose

Provides a fine-tuned BERT classifier for every dataset, trained on SageMaker with validation-only model selection, so that zero-shot models can be compared with a supervised model that reads the same text they do.

## ADDED Requirements

### Requirement: Fine-tuned encoder on every dataset
The study SHALL fine-tune a pinned revision of a pretrained BERT-base encoder with a classification head on each of the five datasets, and for every repeat of the repeated-split datasets, and SHALL produce a probability vector over the dataset's full ordered class set for exactly the frozen evaluation identifiers used for Jev. Tabular rows SHALL be given to the encoder as the same serialized text Jev receives. The encoder's model identifier and revision SHALL be saved with the predictions.

#### Scenario: Same identifiers as Jev
- **WHEN** encoder predictions are read for a dataset
- **THEN** their identifiers equal the frozen evaluation identifiers, every row has a probability for every class in the dataset's class order, and reading fails if any identifier is missing or extra

#### Scenario: Tabular input
- **WHEN** the encoder is trained on German Credit or Breast Cancer Wisconsin Diagnostic
- **THEN** each row is the serialized text used in the Jev request for that row

### Requirement: Encoder selection without test labels
Learning rate, number of epochs and any other training choice SHALL be selected on validation labels only. Evaluation labels SHALL NOT be provided to the training job; metrics SHALL be computed afterwards from the returned probabilities.

#### Scenario: Evaluation labels withheld
- **WHEN** data is prepared for an encoder training job
- **THEN** the evaluation examples are supplied without labels, and the saved run records the configuration selected and the validation score it was selected on

### Requirement: Encoder learning curve
The encoder SHALL be trained on the same nested training fractions as the XGBoost learning curve, with the selected configuration, and scored on the evaluation examples. A fraction that cannot be trained within the job's time limit SHALL be recorded as skipped with the reason.

#### Scenario: Curve produced
- **WHEN** the encoder run completes for a dataset
- **THEN** evaluation accuracy and macro-F1 are saved for each training-set size, with the number of training labels

### Requirement: Encoder jobs on SageMaker GPU
Encoder training SHALL run as a SageMaker pipeline training job on a GPU instance, trying the configured instance types and then the configured regions in order without asking, and recording every attempt, the container version, the execution identifier and the billable seconds. No encoder result SHALL be produced outside SageMaker without that substitution being recorded and stated.

#### Scenario: Run recorded
- **WHEN** an encoder execution completes
- **THEN** its execution identifier, instance type, region, container image, hyperparameters, billable seconds and cost are saved with the predictions

#### Scenario: Instance refused
- **WHEN** the account refuses to start the job on an instance type or in a region
- **THEN** the next configured combination is tried and the refused attempt is recorded with its error
