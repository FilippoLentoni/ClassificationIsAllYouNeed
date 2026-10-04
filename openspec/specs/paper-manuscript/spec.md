# paper-manuscript Specification

## Purpose

Produces the short arXiv-style paper "Classification Is All You Need", which argues that a measure on a finite class space unlocks classical classification methodology, and presents the experimental evidence.

## Requirements

### Requirement: Formal framing
The Background section SHALL formally contrast an autoregressive LM's probability measure on the token-sequence space (Σ*, with the class obtained only through a parsing map that may leak mass to invalid strings) with a measure on the finite measurable space (C, 2^C), and SHALL state which classical tools require the latter.

#### Scenario: Formal statement present
- **WHEN** a reader reads the Background section
- **THEN** it contains the definitions of both measurable spaces, the pushforward through the parsing map, and the invalid-mass term

### Requirement: Generated, traceable results
All numeric tables and figures in the paper SHALL be generated from experiment outputs by a script, never typed by hand, and the paper SHALL build with a single command.

#### Scenario: One-command build
- **WHEN** the build command runs after the experiments
- **THEN** tables and figures are regenerated from the results directory and a PDF is produced without LaTeX errors

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

### Requirement: Prior public evaluations
The paper SHALL cite prior public evaluations of Jev on the datasets it uses. Every number taken from a prior evaluation SHALL be checked against its source before the paper is built, and SHALL be stored with its source reference, the date it was checked and the protocol differences that matter for comparison (model version, split, sample size, prompt). Where no public evaluation of a dataset was found, the paper SHALL say so.

#### Scenario: Verified citation
- **WHEN** the related-evaluations table is generated
- **THEN** every prior result in it has a recorded source, check date and protocol note, and the build fails if any is missing

#### Scenario: No prior evaluation found
- **WHEN** no public Jev evaluation of a dataset was found
- **THEN** the paper states that none was found for that dataset

### Requirement: Zero-shot and supervised results kept distinct
The paper SHALL label Jev and Qwen3.6-27B as zero-shot and XGBoost and BERT as supervised wherever they are compared, and SHALL report, for each dataset and model, the number of labels used for training, for calibration, for threshold selection and for escalation-threshold selection.

#### Scenario: Label budget table
- **WHEN** the paper is compiled
- **THEN** a table lists, for each dataset and model, the training, calibration and threshold-selection label counts, with zero training labels shown for Jev and Qwen

### Requirement: Reported numbers verified against saved outputs
Every number in the compiled paper SHALL be traceable to a saved experiment output, and the build SHALL include an automated check that each reported number equals the value in the saved outputs. The rendered PDF SHALL be inspected before the paper is declared final.

#### Scenario: Number check passes
- **WHEN** the verification step runs after the paper is built
- **THEN** it reports every generated number with its source file and confirms that each matches, or fails and names the mismatches

#### Scenario: Rendered PDF inspected
- **WHEN** the paper is declared final
- **THEN** a record exists that each page of the rendered PDF was inspected for missing figures, overflowing tables and unresolved references

### Requirement: Findings report
The project SHALL include a concise report of the findings, the limitations, the actual Jev and AWS costs, and anything that remains incomplete, with each figure in it taken from the saved outputs.

#### Scenario: Incomplete work stated
- **WHEN** a planned experiment could not be completed
- **THEN** the report names it and gives the reason, and the paper does not report results for it

### Requirement: Qwen prompt examples
The appendix SHALL show, for one evaluation example per dataset, the exact prompt rendered for Qwen and its saved per-class probabilities, copied from the saved outputs.

#### Scenario: Prompt from saved outputs
- **WHEN** the appendix is generated
- **THEN** each Qwen prompt shown is byte-identical to the saved rendered prompt for that example

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
