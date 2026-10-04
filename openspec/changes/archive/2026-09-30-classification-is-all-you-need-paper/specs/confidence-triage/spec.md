# Spec Delta

## Purpose

Quantifies how per-example confidence enables selective automation: examples below a confidence threshold K are escalated to a simulated human reviewer, trading cost against error.

## ADDED Requirements

### Requirement: Confidence-based routing
Given Jev's predictions, the experiment SHALL route each test example with confidence < K to the simulated human reviewer and keep Jev's prediction otherwise, for a sweep of K values covering coverage from 0% to 100%. Confidence SHALL be Jev's maximum class probability, and the Jev-native confidence statistic SHALL be evaluated as an alternative.

#### Scenario: Full sweep
- **WHEN** the triage experiment runs for Jev on German Credit
- **THEN** results are produced for each K in the sweep, including K values giving full automation and full escalation

### Requirement: Escalation target
The only escalation target SHALL be a simulated human-in-the-loop oracle that returns the gold label with a configurable accuracy (default 100%) at a configurable per-item cost. No other model SHALL be used for escalation.

#### Scenario: HITL oracle
- **WHEN** examples are escalated to the simulated human with accuracy 1.0
- **THEN** those examples are counted as correct and each adds the configured human cost

### Requirement: Cost model and optimal K
The experiment SHALL compute total cost as Jev inference cost + human escalation cost + misclassification cost (from the dataset cost matrix), and SHALL report the K minimizing total cost on validation together with its test-set cost, accuracy, macro-F1 and coverage.

#### Scenario: Optimal threshold report
- **WHEN** the sweep completes
- **THEN** a table lists the validation-optimal K and its test-set total cost, coverage, accuracy and macro-F1

### Requirement: Risk–coverage analysis
The experiment SHALL produce a risk–coverage curve (selective error rate versus coverage) for Jev and report the area under the risk–coverage curve (AURC).

#### Scenario: Jev risk–coverage
- **WHEN** the risk–coverage curve is produced for Jev on German Credit
- **THEN** a figure shows the curve together with the random-ranking reference, and a table reports the AURC
