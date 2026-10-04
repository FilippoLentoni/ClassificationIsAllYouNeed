# Spec Delta

## MODIFIED Requirements

### Requirement: Faithfulness evaluation
The experiment SHALL evaluate KernelSHAP and LIME with a deletion test on live Jev: remove the top-k attributed features or words, re-query Jev, and report the drop in predicted-class probability and the label-flip rate, compared against removing k randomly chosen features or words.

#### Scenario: Deletion versus random baseline
- **WHEN** the deletion test runs for KernelSHAP with k = 3 on a dataset
- **THEN** the report gives the mean probability drop and flip rate for top-k deletion and for random-k deletion

### Requirement: Agreement and cost reporting
The experiment SHALL report, per dataset, rank agreement (Kendall's τ and top-3 overlap) between KernelSHAP and LIME, the number of Jev requests each method needed per explanation, and the total Jev requests and cost of the experiment.

#### Scenario: Agreement and cost tables
- **WHEN** KernelSHAP and LIME explanations exist for the sampled examples of a dataset
- **THEN** a table of Kendall's τ and top-3 overlap is written, plus the requests per explanation for each method and the total requests and cost

### Requirement: Mock results are not Jev findings
Every explainability artifact SHALL carry the mode it was produced in, and the live paper SHALL present only explainability results produced by live Jev requests.

#### Scenario: Live paper build
- **WHEN** the paper is built in live mode
- **THEN** every explainability number and figure in it comes from live-mode artifacts, and the build marks the paper as a mock draft if any of them is mock-labelled

## REMOVED Requirements

### Requirement: Perturbation attributions over a class-probability function
**Reason**: That requirement confined attributions to the mock and forbade live Jev calls, because of the earlier 100-inference cap. The cap is raised and explanations now run on Jev.
**Migration**: Replaced by "Perturbation attributions on live Jev". The mock implementation is kept only for tests and offline runs.

## ADDED Requirements

### Requirement: Perturbation attributions on live Jev
The experiment SHALL compute KernelSHAP and LIME attributions of the predicted class's probability on live Jev for a seeded sample of evaluation examples from each of the five datasets, perturbing features (tabular) or words (text) by removing them from the input. The number of perturbations per example SHALL be bounded by configuration, and the Jev requests the experiment would make SHALL be checked against its budget before any is sent.

#### Scenario: Tabular attribution
- **WHEN** KernelSHAP runs on a German Credit or Breast Cancer Wisconsin Diagnostic example for Jev
- **THEN** an attribution value is produced for every serialized feature, and the attributions plus the base value approximately reconstruct Jev's output probability (reported additivity error)

#### Scenario: Text attribution
- **WHEN** LIME runs on an SST-2, AG News or Banking77 example for Jev
- **THEN** a signed weight is produced for each word of the input

#### Scenario: Budget checked first
- **WHEN** the explanation stage would need more Jev requests than its budget or the remaining cap allows
- **THEN** it stops before sending any request and reports the requests required and allowed
