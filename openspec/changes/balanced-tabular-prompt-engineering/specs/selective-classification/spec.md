## Purpose

Shows how a model's accuracy and the share of items it classifies automatically move with a confidence threshold, so that a reader can see what would be left to a person or a stronger model, without selecting a threshold.

## ADDED Requirements

### Requirement: Accuracy and coverage against the confidence threshold
For every model and dataset the study SHALL compute, on the test set, for each confidence threshold on a fixed grid: the coverage, which is the share of test items whose maximum class probability is at or above the threshold, and the accuracy on those items. Items below the threshold are the ones left to a person or a stronger model.

#### Scenario: Curve for every model
- **WHEN** the stage runs for a dataset
- **THEN** a figure shows accuracy and coverage against the threshold for each model that has predictions, and a table gives coverage, accuracy and the number of remaining errors at the configured thresholds

#### Scenario: Supervised models at each training size
- **WHEN** the table is generated for a dataset
- **THEN** it has a row for XGBoost and a row for BERT at each training size (50%, 75% and 100% of the training part), computed from the probabilities of the model trained at that size, so that the sensitivity of accuracy and coverage to the number of training labels can be read

#### Scenario: Lowest threshold covers everything
- **WHEN** the threshold is at or below one over the number of classes
- **THEN** coverage is 1 and the accuracy equals the model's accuracy on the whole test set

#### Scenario: Empty coverage
- **WHEN** no test item reaches a threshold
- **THEN** coverage is 0 and accuracy is reported as not defined, not as zero

### Requirement: No threshold is selected
The study SHALL NOT select, tune or recommend a threshold from these curves or from any labelled data. Decisions reported elsewhere in the study SHALL use the argmax label.

#### Scenario: Descriptive only
- **WHEN** the results and the paper are generated
- **THEN** they contain no validation-selected, cost-optimal or otherwise chosen threshold

### Requirement: Computed from saved predictions
The curves SHALL be computed from saved predictions and SHALL make no Jev request and start no SageMaker job.

#### Scenario: No remote calls
- **WHEN** the stage runs
- **THEN** the Jev ledger and the record of SageMaker executions are unchanged
