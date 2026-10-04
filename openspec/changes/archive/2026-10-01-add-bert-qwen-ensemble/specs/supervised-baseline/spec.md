## MODIFIED Requirements

### Requirement: AWS resource lifecycle and cost
The AWS resources the study needs (for XGBoost, BERT and Qwen) SHALL be defined in one reviewable template that can be deployed in any of the configured regions, SHALL be created or changed only after the user has approved that template, SHALL grant no more access than the study needs, and SHALL be removable with one documented command that covers every region in which they were created, including staged model weights. A spending limit of USD 75 for all SageMaker work of the study SHALL be configured and checked, against the recorded cost of every earlier execution, before any job is started, and the actual cost of every job SHALL be recorded.

#### Scenario: Approval before creation
- **WHEN** the resources, or a change to them, have not been approved by the user
- **THEN** nothing is created or changed in the AWS account

#### Scenario: Spending limit
- **WHEN** the estimated cost of the jobs about to start, added to the cost already recorded, exceeds the configured limit
- **THEN** no job is started and the estimate and the limit are reported

#### Scenario: Teardown
- **WHEN** the documented teardown command is run
- **THEN** the storage bucket with any staged model weights, the execution role and the permissions added for the study are removed in every region where they were created
