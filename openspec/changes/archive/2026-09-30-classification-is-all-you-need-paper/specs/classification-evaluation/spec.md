# Spec Delta

## Purpose

Evaluates Jev's probabilistic outputs with standard classification metrics and calibration diagnostics, showing that the classical toolbox applies directly to a class distribution.

## ADDED Requirements

### Requirement: Classical classification metrics
The evaluation SHALL compute, for Jev on the German Credit Jev test subset, accuracy, per-class precision, recall and F1, macro- and micro-averaged F1, the confusion matrix, and ROC-AUC and PR-AUC (one-vs-rest for multiclass).

#### Scenario: Binary dataset metrics
- **WHEN** evaluation runs on German Credit for Jev
- **THEN** a results table contains accuracy, precision, recall, F1 per class, macro-F1, ROC-AUC, PR-AUC and the 2×2 confusion matrix

### Requirement: Calibration analysis
The evaluation SHALL compute expected calibration error (ECE, with a documented binning scheme), Brier score and log-loss, and SHALL produce a reliability diagram for Jev.

#### Scenario: Reliability diagram
- **WHEN** calibration analysis runs for Jev on German Credit
- **THEN** a reliability diagram figure and a table row with ECE, Brier and log-loss are written to the results directory

### Requirement: Uncertainty reporting
Every headline metric SHALL be reported with a bootstrap 95% confidence interval computed over test examples with a fixed seed.

#### Scenario: Confidence intervals in results
- **WHEN** the metrics table is generated
- **THEN** each metric cell includes its point estimate and 95% bootstrap interval

### Requirement: Invalid-output accounting
The evaluation SHALL report the rate of invalid or missing Jev outputs and SHALL count them as errors in all metrics.

#### Scenario: Missing Jev response
- **WHEN** an item in the Jev subset has no valid Jev response
- **THEN** it is counted as an incorrect prediction and contributes to the reported invalid-output rate
