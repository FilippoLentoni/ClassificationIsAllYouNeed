# model-cost-latency-comparison Specification

## Purpose

Compares Jev, Qwen3.6-27B, BERT and XGBoost on F1, latency and cost as a function of the number of training labels, from measured and saved values only.

## Requirements

### Requirement: F1 by training labels
The study SHALL produce, for each dataset, macro-F1 on the evaluation examples against the number of training labels, with one curve per supervised model and one horizontal line per zero-shot model at zero training labels.

#### Scenario: Figure and table
- **WHEN** the comparison stage runs
- **THEN** a figure and a table give macro-F1 at each training-set size for BERT and XGBoost, and macro-F1 for Jev and Qwen with thinking disabled

### Requirement: Cost per prediction and training cost
The study SHALL report, for each model and dataset, the cost per 1,000 predictions and, for supervised models, the training cost, computed only from recorded quantities: Jev token counts and its published price, and SageMaker billable seconds at the configured list price per instance hour. How each figure is computed SHALL be stated with it.

#### Scenario: Costs traced
- **WHEN** a cost figure is reported
- **THEN** it is traced to the saved token counts or billable seconds it was computed from

#### Scenario: Self-hosted cost includes loading
- **WHEN** Qwen's cost per 1,000 predictions is reported
- **THEN** it is given both over the whole billed job, including model loading, and over inference time only

### Requirement: Latency reported by kind
Latency SHALL be reported as measured, and batch throughput SHALL NOT be presented as per-request latency. For Jev it SHALL be the measured request latency. For Qwen it SHALL be both the batch throughput in items per second and the single-request latency measured on a fixed sample sent one at a time. For BERT and XGBoost it SHALL be the measured batch inference time per item; XGBoost is rerun with this timing because its first runs did not record it.

#### Scenario: Qwen single-request latency
- **WHEN** Qwen latency is reported
- **THEN** the single-request figure comes from a fixed seeded sample of items sent one at a time to the loaded model, and the sample size is stated
