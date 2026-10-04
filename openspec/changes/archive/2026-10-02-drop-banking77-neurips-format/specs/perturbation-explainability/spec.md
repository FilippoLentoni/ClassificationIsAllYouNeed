## MODIFIED Requirements

### Requirement: Perturbation attributions on live Jev
The experiment SHALL compute KernelSHAP and LIME attributions of the predicted class's probability on live Jev for a seeded sample of evaluation examples from each of the four datasets, perturbing features (tabular) or words (text) by removing them from the input. The number of perturbations per example SHALL be bounded by configuration, and the Jev requests the experiment would make SHALL be checked against its budget before any is sent. The saved attributions of the four datasets SHALL be kept as they are; none is recomputed for this change.

#### Scenario: Tabular attribution
- **WHEN** KernelSHAP runs on a German Credit or Breast Cancer Wisconsin Diagnostic example for Jev
- **THEN** an attribution value is produced for every serialized feature, and the attributions plus the base value approximately reconstruct Jev's output probability (reported additivity error)

#### Scenario: Text attribution
- **WHEN** LIME runs on an SST-2 or AG News example for Jev
- **THEN** a signed weight is produced for each word of the input

#### Scenario: Budget checked first
- **WHEN** the explanation stage would need more Jev requests than its budget or the remaining cap allows
- **THEN** it stops before sending any request and reports the requests required and allowed
