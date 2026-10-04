# Spec Delta

## Purpose

Produces the short arXiv-style paper "Classification Is All You Need", which argues that a measure on a finite class space unlocks classical classification methodology, and presents the experimental evidence.

## ADDED Requirements

### Requirement: Paper structure
The manuscript SHALL be a LaTeX paper of about 6–8 pages (excluding references and appendix) titled "Classification Is All You Need", with sections: Abstract, Introduction, Background (from a σ-algebra over token sequences to a σ-algebra over classes), Setup (datasets, models, protocol), one results section for each of the five techniques (metrics & calibration, cost-sensitive thresholding, triage/HITL, importance-sampling evaluation, perturbation explainability, presented as method and call cost only), Limitations, Conclusion, and References.

#### Scenario: Section completeness
- **WHEN** the paper is compiled
- **THEN** the PDF contains every listed section, and each results section except explainability references at least one generated table or figure built from live Jev results

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
The paper SHALL state the Jev model version, the date of the runs, sample sizes, the total number of Jev inferences used (at most 100), and cost/latency measurements. It SHALL NOT be compiled as a final version while any included result is mock-labelled, and SHALL NOT present mock explainability results as Jev findings.

#### Scenario: Mock results guard
- **WHEN** the paper is built from results that include any mock-labelled artifact
- **THEN** the PDF carries a visible "DRAFT — MOCK RESULTS" watermark
