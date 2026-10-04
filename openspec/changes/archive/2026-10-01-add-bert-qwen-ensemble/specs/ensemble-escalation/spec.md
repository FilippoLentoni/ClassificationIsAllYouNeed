## Purpose

Measures what happens to F1 when Jev's low-confidence predictions are escalated to Qwen3.6-27B, using saved predictions only and a threshold chosen without evaluation labels.

## ADDED Requirements

### Requirement: Offline escalation from saved predictions
The escalation ensemble SHALL take Qwen's prediction (thinking disabled) for every item whose Jev maximum class probability is below a threshold, and Jev's prediction otherwise. It SHALL be computed from saved Jev and Qwen predictions and SHALL make no Jev request and start no SageMaker job.

#### Scenario: No remote calls
- **WHEN** the escalation stage runs
- **THEN** the Jev ledger and the record of SageMaker executions are unchanged

### Requirement: Escalation curve
For each dataset the stage SHALL report macro-F1 and accuracy of the ensemble against the share of evaluation items escalated, over every distinct Jev confidence value, with Jev alone and Qwen alone as the two endpoints, and the cost and mean latency implied by each share.

#### Scenario: Curve endpoints
- **WHEN** the escalation curve is computed
- **THEN** escalating no item gives Jev's results and escalating every item gives Qwen's results

### Requirement: Validation-selected escalation threshold
The reported operating point SHALL use a threshold selected on validation items only, maximising macro-F1 with ties broken towards fewer escalations. The validation items are each repeat's validation part for the tabular datasets, Jev's threshold-selection items for SST-2, and 500 seeded items of the validation part for AG News and Banking77, which both Jev and Qwen predict for this purpose. Evaluation labels SHALL NOT influence the threshold.

#### Scenario: Labels hidden
- **WHEN** the evaluation labels are permuted
- **THEN** the selected threshold is unchanged

#### Scenario: Validation items for the multiclass datasets
- **WHEN** the escalation threshold is selected for AG News or Banking77
- **THEN** it is selected on the 500 seeded validation items, none of which is an evaluation item

### Requirement: Escalation ceiling stated
The stage SHALL report, for each dataset, the number of Jev errors whose confidence equals the largest reported value, which no threshold can escalate without escalating every item at that confidence.

#### Scenario: Unreachable errors counted
- **WHEN** the escalation results are saved
- **THEN** they include the count of Jev errors at the maximum reported confidence and the share of items at that confidence
