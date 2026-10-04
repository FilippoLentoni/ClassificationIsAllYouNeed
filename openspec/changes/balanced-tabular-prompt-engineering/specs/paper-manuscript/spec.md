## MODIFIED Requirements

### Requirement: Zero-shot and supervised results kept distinct
The paper SHALL label Jev with its original prompt and Qwen3.6-27B as zero-shot and XGBoost and BERT as supervised wherever they are compared, SHALL label Jev with the engineered prompt as prompt-engineered on training data and never as zero-shot, and SHALL report, for each dataset and model, the number of labels used for training, for model selection and for prompt engineering.

#### Scenario: Label budget table
- **WHEN** the paper is compiled
- **THEN** a table lists, for each dataset and model, the training, model-selection and prompt-engineering label counts, with zero labels shown for zero-shot Jev and for Qwen

#### Scenario: Engineered prompt labelled
- **WHEN** Jev with the engineered prompt appears in a table, figure or sentence
- **THEN** it is named as prompt-engineered and its row in the label table shows the training and validation labels it used

## REMOVED Requirements

### Requirement: Paper structure within the submission format
**Reason**: It listed a threshold-selection section and forbade dropping any earlier result; threshold selection and the imbalanced tabular results are now removed at the user's decision.
**Migration**: Replaced by "Paper structure for the balanced study".

## ADDED Requirements

### Requirement: Paper structure for the balanced study
The manuscript SHALL be a LaTeX paper titled "Classification Is All You Need" in the NeurIPS 2026 submission format (capability `submission-format`). Its main text, at most nine pages including figures and tables, SHALL contain: Abstract, Introduction, Background, Related evaluations of Jev, Setup (datasets with the balanced tabular splits, models, protocol), results on metrics and calibration, the comparison of F1, cost and latency by training labels, accuracy and coverage against the confidence threshold, prompt engineering for Jev, escalation of low-confidence Jev predictions to Qwen as a curve, importance-sampling evaluation and perturbation explanations, then Limitations and Conclusion, followed by References, Appendices and the checklist. It SHALL NOT contain a threshold-selection section, a selected threshold, a decision-cost table, or results for the imbalanced tabular splits. It SHALL state which results (importance sampling, explanations) were computed with the original prompt on the original rows.

#### Scenario: Section completeness
- **WHEN** the paper is compiled
- **THEN** the PDF contains every listed section, and each results section references at least one generated table or figure built from live results, in the main text or in an appendix it cites

#### Scenario: No selected threshold
- **WHEN** the paper is compiled
- **THEN** it reports no validation-selected or cost-based threshold and no decision cost

#### Scenario: Error counts for the tabular datasets
- **WHEN** the tabular results are shown
- **THEN** a table gives true positives, false negatives, false positives and true negatives at the argmax decision for every model entry on the balanced test parts

### Requirement: Abstract shown before the rebuild
Because the abstract was written by the user and its threshold-selection and escalation sentences no longer hold, the rewritten abstract and the numbers behind it SHALL be shown to the user before the final paper is built.

#### Scenario: Review before building
- **WHEN** the new results are complete
- **THEN** the proposed abstract text and its numbers are presented to the user, and the final build uses the text the user confirms
