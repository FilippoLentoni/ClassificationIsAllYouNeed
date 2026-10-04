# Spec Delta

## MODIFIED Requirements

### Requirement: Paper structure
The manuscript SHALL be a LaTeX paper titled "Classification Is All You Need" with a main body of about 8–10 pages (excluding references and appendix) and these sections: Abstract, Introduction, Background (from a σ-algebra over token sequences to a σ-algebra over classes), Related evaluations of Jev, Setup (datasets, models, protocol), one results section for each of metrics and calibration against the supervised baseline, threshold selection, importance-sampling evaluation and perturbation explanations, then Limitations, Conclusion, References, and an Appendix holding the request and response examples, the dataset documentation and the per-class tables. It SHALL NOT contain a triage section.

#### Scenario: Section completeness
- **WHEN** the paper is compiled
- **THEN** the PDF contains every listed section, and each results section references at least one generated table or figure built from live results

#### Scenario: All five datasets reported
- **WHEN** the paper is compiled
- **THEN** the metrics section reports Jev and the supervised baseline on each of the five datasets

### Requirement: Honest provenance
The paper SHALL state the Jev model version and SDK version, the date of the runs, the dataset versions and sample sizes, the total number of Jev inferences used and their actual cost and latency computed over all of them, and the actual AWS cost of the baseline. It SHALL state which earlier requests predate request logging. It SHALL NOT be compiled as a final version while any included result is mock-labelled, and SHALL NOT state a measurement that is not present in the saved outputs.

#### Scenario: Mock results guard
- **WHEN** the paper is built from results that include any mock-labelled artifact
- **THEN** the PDF carries a visible "DRAFT — MOCK RESULTS" watermark

#### Scenario: Costs cover all requests
- **WHEN** the paper reports the number of Jev inferences and their cost
- **THEN** the cost is computed over exactly the inferences counted, including template checks, the smoke test and failed requests

## ADDED Requirements

### Requirement: Exact request and response examples
The paper SHALL include five complete Jev exchanges, one from each dataset, taken verbatim from the request and response log. Each SHALL show the serialized input, the exact instructions, every label description in the order sent, the requested and reported model versions, the SDK version, and from the response the selected label, the complete probability vector and the reported confidence. The paper SHALL describe missing-value handling in the serialization. An exchange that was not logged when it happened SHALL NOT be shown as an original; if shown at all it SHALL be labelled as regenerated from code, with the result of checking it against the stored cache key.

#### Scenario: Examples come from the log
- **WHEN** the appendix is generated
- **THEN** each of the five exchanges is copied from a logged record, and the generator fails if a dataset has no logged exchange

#### Scenario: Complete probability vector
- **WHEN** the Banking77 example is shown
- **THEN** all 77 label descriptions and all 77 probabilities appear, in the order sent

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
The paper SHALL label Jev as zero-shot and the baseline as supervised wherever they are compared, and SHALL report, for each dataset and model, the number of labels used for training, for calibration and for threshold selection.

#### Scenario: Label budget table
- **WHEN** the paper is compiled
- **THEN** a table lists, for each dataset and model, the training, calibration and threshold-selection label counts, with zero training labels shown for Jev

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
