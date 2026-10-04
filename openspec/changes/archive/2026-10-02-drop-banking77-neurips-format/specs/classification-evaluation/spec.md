## MODIFIED Requirements

### Requirement: Classical classification metrics
The evaluation SHALL compute, for every (dataset, model) pair on that dataset's frozen evaluation examples, accuracy, macro-F1, per-class precision, recall and F1, and the confusion matrix. For binary datasets it SHALL also compute ROC-AUC and PR-AUC for the documented positive class. For multiclass datasets it SHALL compute one-vs-rest macro-averaged ROC-AUC and PR-AUC over the classes present in the evaluation examples. Jev, Qwen3.6-27B, XGBoost and BERT SHALL be scored on the identical examples; Qwen results with thinking enabled SHALL be scored separately on the subsample they cover and compared with the other models on that same subsample only.

#### Scenario: Binary dataset metrics
- **WHEN** evaluation runs on German Credit, Breast Cancer Wisconsin Diagnostic or SST-2 for a model
- **THEN** the results contain accuracy, macro-F1, precision, recall and F1 per class, ROC-AUC and PR-AUC for the positive class, and the 2×2 confusion matrix

#### Scenario: Multiclass dataset metrics
- **WHEN** evaluation runs on AG News for a model
- **THEN** the results contain accuracy, macro-F1, precision, recall and F1 for every class, and one-vs-rest macro ROC-AUC and PR-AUC

#### Scenario: Same examples for both models
- **WHEN** metrics are computed for any two of the four models on a dataset
- **THEN** both are computed on the same example identifiers, and the evaluation stops with an error if the identifier sets differ

#### Scenario: Thinking subsample compared on its own items
- **WHEN** Qwen with thinking enabled is reported for a dataset
- **THEN** the other models' metrics shown next to it are computed on the same subsample identifiers

### Requirement: Uncertainty reporting
Every headline metric SHALL be reported with its uncertainty. For datasets evaluated on an official held-out split, this SHALL be a bootstrap 95% confidence interval over evaluation examples with a fixed seed. For datasets evaluated on repeated stratified splits, this SHALL be the mean and standard deviation across the repeats, with the number of repeats stated.

#### Scenario: Confidence intervals in results
- **WHEN** the metrics table is generated for SST-2 or AG News
- **THEN** each headline metric includes its point estimate and 95% bootstrap interval

#### Scenario: Variation across repeated splits
- **WHEN** the metrics table is generated for German Credit or Breast Cancer Wisconsin Diagnostic
- **THEN** each headline metric includes its mean and standard deviation across the split repeats and the number of repeats
