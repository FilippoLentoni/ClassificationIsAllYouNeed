## MODIFIED Requirements

### Requirement: SageMaker pipeline orchestration
Baseline preprocessing and training SHALL run as a SageMaker pipeline with a processing job and a training job for each dataset. The pipeline definition, the job parameters, the container versions and the execution identifiers SHALL be recorded. When the preferred instance type or region cannot run the jobs, the stage SHALL try the configured alternative instance types and then the configured alternative regions, in order, without asking, recording every attempt and its outcome. Only when every configured combination has failed SHALL the stage stop and report, and no baseline result SHALL be produced by a route other than SageMaker without that substitution being recorded and stated.

#### Scenario: Pipeline run recorded
- **WHEN** a baseline pipeline execution completes for a dataset
- **THEN** its execution identifier, job names, instance types, container versions, hyperparameters and billable seconds are saved with the predictions

#### Scenario: Instance type or region not available
- **WHEN** the account refuses to start a job on an instance type or in a region (for example zero quota or no capacity)
- **THEN** the stage moves to the next configured instance type, and after the instance types are exhausted to the next configured region, and records the refused attempt with its error

#### Scenario: All datasets on one combination
- **WHEN** a working instance type and region are found
- **THEN** all four datasets are run on that combination, and the instance type and region used are saved with the predictions and stated in the paper

#### Scenario: SageMaker unavailable
- **WHEN** every configured instance type in every configured region has refused the jobs
- **THEN** the stage stops, reports every attempt and its error, and produces no baseline predictions

#### Scenario: Reproducible definition
- **WHEN** the pipeline definition is generated twice from the same configuration
- **THEN** the two definitions are identical
