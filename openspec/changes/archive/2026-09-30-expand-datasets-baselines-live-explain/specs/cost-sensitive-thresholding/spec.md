# Spec Delta

## MODIFIED Requirements

### Requirement: Cost matrix specification
The experiment SHALL accept, for each binary dataset, a cost matrix C[true, predicted] from configuration. German Credit SHALL default to its canonical matrix (cost 5 for predicting "good" when the true class is "bad", 1 for the reverse, 0 on the diagonal). Any cost matrix that is not canonical for its dataset SHALL be labelled as an illustrative assumption wherever it is reported.

#### Scenario: German Credit default costs
- **WHEN** the thresholding experiment runs on German Credit without overrides
- **THEN** the canonical 5:1 cost matrix is used and recorded in the results

#### Scenario: Illustrative costs are labelled
- **WHEN** a dataset without a canonical cost matrix is reported
- **THEN** its cost matrix is recorded in the results and marked as an illustrative assumption

### Requirement: Bayes-optimal reference and baselines
For each binary dataset and each model, the experiment SHALL apply the decision rule "predict the positive class when P(positive | x) ≥ t" and SHALL report, on the evaluation examples, precision, recall, F1 for the positive class, accuracy and mean decision cost at three thresholds: (a) the default t = 0.5, (b) the analytic cost threshold C_FP / (C_FP + C_FN), and (c) the validation-selected threshold.

#### Scenario: Three thresholds reported
- **WHEN** the thresholding experiment completes for a binary dataset and a model
- **THEN** a table lists the value of each of the three thresholds and, at each, precision, recall, positive-class F1, accuracy and mean decision cost on the evaluation examples

#### Scenario: Binary sanity check
- **WHEN** the thresholding experiment completes for a binary dataset and a model
- **THEN** the report includes the validation-selected threshold alongside the analytic threshold C_FP / (C_FP + C_FN) and the absolute difference between them

#### Scenario: Threshold rule at the boundary
- **WHEN** an example has P(positive | x) exactly equal to t
- **THEN** it is predicted positive

### Requirement: Cost-curve visualization
For each binary dataset and each model, the experiment SHALL produce a figure with two separate panels over a sweep of t from 0 to 1: one panel showing precision, recall, accuracy and positive-class F1 against t, and one panel showing mean decision cost against t. If macro-F1 is shown, it SHALL be a separately labelled curve. Both panels SHALL mark the default threshold, the analytic cost threshold and the validation-selected threshold.

#### Scenario: Figure generation
- **WHEN** the thresholding experiment completes for a binary dataset
- **THEN** a two-panel figure is written for each model, with the metric curves in one panel, decision cost in the other, and the three thresholds marked and labelled in both

## REMOVED Requirements

### Requirement: Genetic-algorithm threshold optimization
**Reason**: Thresholding is simplified to a single threshold on P(positive | x); a one-dimensional threshold is selected directly on validation data, so a genetic algorithm adds nothing.
**Migration**: Use the validation-selected threshold defined by "Validation-selected threshold". Remove the genetic algorithm, its configuration and its tests.

## ADDED Requirements

### Requirement: Validation-selected threshold
The experiment SHALL select, for each binary dataset and model, the threshold t that minimizes mean decision cost on validation examples only, using a documented deterministic tie-break, and SHALL evaluate it on the evaluation examples. Evaluation labels SHALL NOT be used to select it. The number of validation labels used SHALL be recorded.

#### Scenario: Validation-only selection
- **WHEN** the threshold is selected
- **THEN** only validation predictions and labels are used, and the result records how many validation labels were used

#### Scenario: Deterministic selection
- **WHEN** selection runs twice on the same validation predictions
- **THEN** it returns the same threshold

#### Scenario: Repeated splits
- **WHEN** the dataset uses repeated stratified splits
- **THEN** a threshold is selected on each repeat's validation part and evaluated on that repeat's test part, and the mean and standard deviation across repeats are reported
