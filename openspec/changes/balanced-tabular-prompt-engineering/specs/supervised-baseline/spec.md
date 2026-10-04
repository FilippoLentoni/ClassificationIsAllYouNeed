## MODIFIED Requirements

### Requirement: Learning curve
The baseline SHALL be trained on 50%, 75% and 100% of the training part, as nested stratified subsets, with the selected hyperparameters and scored on the evaluation examples, producing a learning curve for each dataset. If the curve cannot be produced for a dataset, that SHALL be recorded with the reason.

#### Scenario: Curve produced
- **WHEN** the learning-curve run completes for a dataset
- **THEN** a figure and a table give evaluation accuracy and macro-F1 at each training-set size, with Jev's zero-shot result as a reference line

#### Scenario: Curve not feasible
- **WHEN** the learning curve cannot be produced for a dataset
- **THEN** the results record that it was skipped and why, and the paper does not show a curve for that dataset

#### Scenario: No size below half
- **WHEN** the learning-curve sizes are generated for a dataset
- **THEN** the smallest training size is half of the training part, and BERT is trained on the same three subsets as XGBoost
