## REMOVED Requirements

### Requirement: Uncertainty reporting
**Reason**: The tabular datasets no longer have split repeats to average over.
**Migration**: Replaced by "Bootstrap uncertainty on every test set".

## ADDED Requirements

### Requirement: Bootstrap uncertainty on every test set
Every headline metric SHALL be reported with a bootstrap 95% confidence interval over the test examples with a fixed seed, for all four datasets. The number of test examples SHALL be shown with the results.

#### Scenario: Confidence intervals in results
- **WHEN** the metrics table is generated for any dataset
- **THEN** each headline metric includes its point estimate and 95% bootstrap interval, and the table states the number of test examples

#### Scenario: Small test sets are visible
- **WHEN** results for the balanced tabular test parts are shown
- **THEN** their test sizes appear next to the intervals, so that wide intervals can be read against them
