## MODIFIED Requirements

### Requirement: Exact request and response examples
The paper SHALL include four complete Jev exchanges, one from each dataset, taken verbatim from the request and response log. Each SHALL show the serialized input, the exact instructions, every label description in the order sent, the requested and reported model versions, the SDK version, and from the response the selected label, the complete probability vector and the reported confidence. The paper SHALL describe missing-value handling in the serialization. An exchange that was not logged when it happened SHALL NOT be shown as an original; if shown at all it SHALL be labelled as regenerated from code, with the result of checking it against the stored cache key.

#### Scenario: Examples come from the log
- **WHEN** the appendix is generated
- **THEN** each of the four exchanges is copied from a logged record, and the generator fails if a dataset has no logged exchange

#### Scenario: Complete probability vector
- **WHEN** the AG News example is shown
- **THEN** all four label descriptions and all four probabilities appear, in the order sent

#### Scenario: Historical request labelled
- **WHEN** a request from before logging existed is shown
- **THEN** it is labelled as regenerated, not as the original payload

### Requirement: Honest provenance
The paper SHALL state the Jev model version and SDK version, the Qwen3.6-27B and BERT model revisions, serving and training container versions, the date of the runs, the dataset versions and sample sizes, the total number of Jev inferences used and their actual cost and latency computed over all of them, and the actual AWS cost and GPU time of every SageMaker job, by model. It SHALL state that a fifth dataset, Banking77, was run and then removed from the study, with the Jev requests and the Jev and AWS cost that belong to it. It SHALL state which earlier requests predate request logging and that Qwen inference ran in training jobs because of the processing-job quota. It SHALL NOT be compiled as a final version while any included result is mock-labelled, and SHALL NOT state a measurement that is not present in the saved outputs.

#### Scenario: Mock results guard
- **WHEN** the paper is built from results that include any mock-labelled artifact
- **THEN** the PDF carries a visible "DRAFT — MOCK RESULTS" watermark

#### Scenario: Costs cover all requests
- **WHEN** the paper reports the number of Jev inferences and their cost
- **THEN** the cost is computed over exactly the inferences counted, including template checks, the smoke test and failed requests

#### Scenario: AWS cost by model
- **WHEN** the paper reports AWS cost
- **THEN** it gives the cost and billable GPU or CPU time for XGBoost, BERT and Qwen separately, including stopped and failed executions

#### Scenario: Removed dataset disclosed
- **WHEN** the paper reports the Jev and AWS totals
- **THEN** it states the share of each that was spent on the removed dataset, without naming anything that identifies the authors

## ADDED Requirements

### Requirement: Paper structure within the submission format
The manuscript SHALL be a LaTeX paper titled "Classification Is All You Need" in the NeurIPS 2026 submission format (capability `submission-format`). Its main text, at most nine pages including figures and tables, SHALL contain: Abstract, Introduction, Background (from a σ-algebra over token sequences to a σ-algebra over classes), Related evaluations of Jev, Setup (datasets, models, protocol), results on metrics and calibration for the four models, the comparison of F1, cost and latency by training labels, escalation of low-confidence Jev predictions to Qwen, threshold selection, importance-sampling evaluation and perturbation explanations, then Limitations and Conclusion. It SHALL be followed by References, Appendices and the NeurIPS paper checklist. Results that do not fit the main text SHALL move to the appendices and be cited from the main text; no result that the previous version reported for the four remaining datasets SHALL be dropped from the paper. The main text SHALL be readable without the appendices. The escalation section SHALL be the only section about routing predictions between models.

#### Scenario: Section completeness
- **WHEN** the paper is compiled
- **THEN** the PDF contains every listed section, and each results section references at least one generated table or figure built from live results, in the main text or in an appendix it cites

#### Scenario: All four datasets reported
- **WHEN** the paper is compiled
- **THEN** the metrics section reports Jev, Qwen3.6-27B, XGBoost and BERT on each of the four datasets, and Banking77 appears nowhere except in the accounting of what was spent on it

#### Scenario: Results moved, not dropped
- **WHEN** a table or figure of the previous version covering the four remaining datasets is not in the main text
- **THEN** it is in an appendix and the main text refers to it

## REMOVED Requirements

### Requirement: Paper structure
**Reason**: It fixed a 10-12 page article layout and five datasets; the paper now follows the NeurIPS 2026 format with four datasets, and its scenario "All five datasets reported" no longer applies.
**Migration**: Replaced by "Paper structure within the submission format" in this capability, which keeps the section list and the rule that the escalation section is the only one about routing, and by the `submission-format` capability for length and layout.
