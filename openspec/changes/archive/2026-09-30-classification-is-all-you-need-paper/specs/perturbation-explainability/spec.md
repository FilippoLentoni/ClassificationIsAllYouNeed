# Spec Delta

## Purpose

Shows that a model returning class probabilities supports classical perturbation-based explanations (KernelSHAP, LIME) and a faithfulness test, with nothing but a class-probability function. Live Jev attributions need about 200 calls per example, far beyond the 100-inference cap, so the pipeline is implemented and validated on mock-jev and makes no live Jev calls.

## ADDED Requirements

### Requirement: Perturbation attributions over a class-probability function
The experiment SHALL compute KernelSHAP and LIME attributions of the predicted class's probability for a seeded sample of German Credit test examples, perturbing features by removing them from the serialization, with the number of perturbations per example bounded by configuration. It SHALL run on mock-jev only and SHALL make zero live Jev calls.

#### Scenario: Tabular attribution
- **WHEN** KernelSHAP runs on a German Credit example for mock-jev
- **THEN** an attribution value is produced for every serialized feature, and the attributions plus the base value approximately reconstruct the model's output probability (reported additivity error)

#### Scenario: No live calls
- **WHEN** the explain stage runs in live mode
- **THEN** it makes zero Jev requests and the Jev ledger total is unchanged

### Requirement: Faithfulness evaluation
The experiment SHALL evaluate KernelSHAP and LIME with a deletion test: remove the top-k attributed features, re-query the same model, and report the drop in predicted-class probability and the label-flip rate, compared against removing k randomly chosen features.

#### Scenario: Deletion versus random baseline
- **WHEN** the deletion test runs for KernelSHAP with k = 3
- **THEN** the report gives the mean probability drop and flip rate for top-k deletion and for random-k deletion

### Requirement: Agreement and cost reporting
The experiment SHALL report rank agreement (Kendall's τ and top-3 overlap) between KernelSHAP and LIME, and the number of model calls each method needs per explanation.

#### Scenario: Agreement and cost tables
- **WHEN** KernelSHAP and LIME explanations exist for the sampled examples
- **THEN** a table of Kendall's τ and top-3 overlap is written, plus a per-method table of model calls per explanation

### Requirement: Mock results are not Jev findings
Every explainability artifact SHALL be mock-labelled, and the live paper SHALL NOT present any explainability number other than the per-explanation call counts as a Jev result.

#### Scenario: Live paper build
- **WHEN** the paper is built in live mode
- **THEN** its explainability section contains the method and the per-explanation call cost, and no mock-derived attribution or faithfulness numbers
