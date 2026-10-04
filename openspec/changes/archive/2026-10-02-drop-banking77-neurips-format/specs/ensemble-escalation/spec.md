## MODIFIED Requirements

### Requirement: Validation-selected escalation threshold
The reported operating point SHALL use a threshold selected on validation items only, maximising macro-F1 with ties broken towards fewer escalations. The validation items are each repeat's validation part for the tabular datasets, Jev's threshold-selection items for SST-2, and 500 seeded items of the validation part for AG News, which both Jev and Qwen predicted for this purpose. Evaluation labels SHALL NOT influence the threshold.

#### Scenario: Labels hidden
- **WHEN** the evaluation labels are permuted
- **THEN** the selected threshold is unchanged

#### Scenario: Validation items for the multiclass datasets
- **WHEN** the escalation threshold is selected for AG News
- **THEN** it is selected on the 500 seeded validation items, none of which is an evaluation item
