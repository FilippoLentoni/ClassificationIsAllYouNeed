# Spec Delta

## Purpose

Demonstrates that class probabilities enable label-efficient, lower-variance performance estimation: labels are acquired with a model-informed proposal distribution and corrected with importance weights.

## ADDED Requirements

### Requirement: Label-budgeted estimation
Given an unlabeled evaluation pool (the 60-item German Credit Jev test subset) and a label budget n smaller than the pool, the experiment SHALL estimate Jev's accuracy (and macro-F1 via a ratio estimator) by sampling n items to label, for both (a) uniform sampling and (b) importance sampling with a proposal q proportional to a smoothed function of the model's expected loss (1 − max class probability, mixed with uniform so that q > 0 everywhere).

#### Scenario: Unbiased weighting
- **WHEN** importance sampling selects item i with probability q_i
- **THEN** its contribution to the estimate is weighted by p_i / q_i where p is the uniform target distribution over the pool

### Requirement: Estimator quality comparison
The experiment SHALL repeat estimation over at least 500 seeded trials per budget for several budgets and SHALL report, per estimator, bias, variance, RMSE against the full-pool ground truth, and empirical 95% interval coverage.

#### Scenario: Variance reduction report
- **WHEN** trials complete for a budget
- **THEN** a table reports bias, variance, RMSE and interval coverage for uniform and importance sampling, and the variance ratio between them

### Requirement: Budget–error visualization
The experiment SHALL produce a figure of RMSE versus label budget for each estimator.

#### Scenario: Figure output
- **WHEN** the experiment finishes
- **THEN** an RMSE-vs-budget figure with one line per estimator is written to the results directory

### Requirement: No additional remote calls
The experiment SHALL reuse cached test-set predictions and SHALL NOT require additional model API calls.

#### Scenario: Offline execution
- **WHEN** the importance-sampling experiment runs with a warm prediction cache
- **THEN** it completes with zero remote calls
