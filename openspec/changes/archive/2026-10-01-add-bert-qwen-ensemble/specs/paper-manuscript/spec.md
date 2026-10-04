## MODIFIED Requirements

### Requirement: Paper structure
The manuscript SHALL be a LaTeX paper titled "Classification Is All You Need" with a main body of about 10–12 pages (excluding references and appendix) and these sections: Abstract, Introduction, Background (from a σ-algebra over token sequences to a σ-algebra over classes), Related evaluations of Jev, Setup (datasets, models, protocol), one results section for each of metrics and calibration against the other models, the comparison of F1, cost and latency by training labels, escalation of low-confidence Jev predictions to Qwen, threshold selection, importance-sampling evaluation and perturbation explanations, then Limitations, Conclusion, References, and an Appendix holding the request and response examples, the Qwen prompt examples, the dataset documentation and the per-class tables. The escalation section SHALL be the only section about routing predictions between models.

#### Scenario: Section completeness
- **WHEN** the paper is compiled
- **THEN** the PDF contains every listed section, and each results section references at least one generated table or figure built from live results

#### Scenario: All five datasets reported
- **WHEN** the paper is compiled
- **THEN** the metrics section reports Jev, Qwen3.6-27B, XGBoost and BERT on each of the five datasets, or marks a missing model result as not available

### Requirement: Honest provenance
The paper SHALL state the Jev model version and SDK version, the Qwen3.6-27B and BERT model revisions, serving and training container versions, the date of the runs, the dataset versions and sample sizes, the total number of Jev inferences used and their actual cost and latency computed over all of them, and the actual AWS cost and GPU time of every SageMaker job, by model. It SHALL state which earlier requests predate request logging and that Qwen inference ran in training jobs because of the processing-job quota. It SHALL NOT be compiled as a final version while any included result is mock-labelled, and SHALL NOT state a measurement that is not present in the saved outputs.

#### Scenario: Mock results guard
- **WHEN** the paper is built from results that include any mock-labelled artifact
- **THEN** the PDF carries a visible "DRAFT — MOCK RESULTS" watermark

#### Scenario: Costs cover all requests
- **WHEN** the paper reports the number of Jev inferences and their cost
- **THEN** the cost is computed over exactly the inferences counted, including template checks, the smoke test and failed requests

#### Scenario: AWS cost by model
- **WHEN** the paper reports AWS cost
- **THEN** it gives the cost and billable GPU or CPU time for XGBoost, BERT and Qwen separately, including stopped and failed executions

### Requirement: Zero-shot and supervised results kept distinct
The paper SHALL label Jev and Qwen3.6-27B as zero-shot and XGBoost and BERT as supervised wherever they are compared, and SHALL report, for each dataset and model, the number of labels used for training, for calibration, for threshold selection and for escalation-threshold selection.

#### Scenario: Label budget table
- **WHEN** the paper is compiled
- **THEN** a table lists, for each dataset and model, the training, calibration and threshold-selection label counts, with zero training labels shown for Jev and Qwen

## ADDED Requirements

### Requirement: Qwen prompt examples
The appendix SHALL show, for one evaluation example per dataset, the exact prompt rendered for Qwen and its saved per-class probabilities, copied from the saved outputs.

#### Scenario: Prompt from saved outputs
- **WHEN** the appendix is generated
- **THEN** each Qwen prompt shown is byte-identical to the saved rendered prompt for that example
