# Spec Delta

## REMOVED Requirements

### Requirement: Confidence-based routing
**Reason**: The confidence-triage experiment is dropped from the study; the revised paper covers metrics, baselines, thresholding, importance sampling and explanations.
**Migration**: None. Remove the triage stage, its configuration, tests, results and paper section. Confidence ranking quality is still visible through calibration analysis in `classification-evaluation`.

### Requirement: Escalation target
**Reason**: There is no escalation once the triage experiment is removed.
**Migration**: None. The simulated human reviewer and its cost settings are deleted.

### Requirement: Cost model and optimal K
**Reason**: The triage cost model is removed with the experiment.
**Migration**: None. Decision cost under asymmetric errors remains in `cost-sensitive-thresholding`.

### Requirement: Risk–coverage analysis
**Reason**: Risk–coverage curves were reported only as part of the triage experiment.
**Migration**: None. Remove the curve, its table entries and its figure from the paper.
