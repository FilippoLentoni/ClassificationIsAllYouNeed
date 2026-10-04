# Spec Delta

## Purpose

Shows that class probabilities allow decision rules tuned to asymmetric misclassification costs, using a genetic algorithm to optimize decision thresholds and an analytic Bayes-optimal rule as a reference.

## ADDED Requirements

### Requirement: Cost matrix specification
The experiment SHALL accept a cost matrix C[true, predicted] from configuration, defaulting to the canonical German Credit matrix (cost 5 for predicting "good" when the true class is "bad", 1 for the reverse, 0 on the diagonal).

#### Scenario: German Credit default costs
- **WHEN** the thresholding experiment runs on German Credit without overrides
- **THEN** the canonical 5:1 cost matrix is used and printed in the results

### Requirement: Genetic-algorithm threshold optimization
The experiment SHALL optimize decision parameters (one threshold for binary tasks; per-class weights/thresholds for multiclass) with a seeded genetic algorithm that minimizes mean misclassification cost on the validation split, and SHALL evaluate the resulting rule on the test split.

#### Scenario: Seeded reproducibility
- **WHEN** the GA runs twice with the same seed and inputs
- **THEN** it returns identical decision parameters and test cost

#### Scenario: Validation-only fitting
- **WHEN** the GA is fitted
- **THEN** only validation-split predictions and labels are used, and test labels are used only for the final report

### Requirement: Bayes-optimal reference and baselines
For Jev on German Credit the experiment SHALL report test-set mean cost for (a) argmax decisions, (b) the analytic Bayes-optimal decision rule applied to the model's probabilities, and (c) the GA-optimized rule, together with precision, recall and F1 under each rule.

#### Scenario: Binary sanity check
- **WHEN** the GA runs on a binary task with calibrated probabilities
- **THEN** the report includes the GA threshold alongside the analytic threshold C_FP / (C_FP + C_FN) and the absolute difference between them

### Requirement: Cost-curve visualization
The experiment SHALL produce a figure of expected cost versus threshold for Jev, marking the argmax, Bayes-optimal and GA operating points.

#### Scenario: Figure generation
- **WHEN** the thresholding experiment completes
- **THEN** a cost-curve figure is written with the three marked operating points
