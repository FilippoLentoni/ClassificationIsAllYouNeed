## MODIFIED Requirements

### Requirement: Classical classification metrics
The evaluation SHALL compute, for every (dataset, model) pair on that dataset's frozen evaluation examples, accuracy, macro-F1, per-class precision, recall and F1, and the confusion matrix. For binary datasets it SHALL also compute ROC-AUC and PR-AUC for the documented positive class. For multiclass datasets it SHALL compute one-vs-rest macro-averaged ROC-AUC and PR-AUC over the classes present in the evaluation examples. Jev, Qwen3.6-27B, XGBoost and BERT SHALL be scored on the identical examples; Qwen results with thinking enabled SHALL be scored separately on the subsample they cover and compared with the other models on that same subsample only.

#### Scenario: Binary dataset metrics
- **WHEN** evaluation runs on German Credit, Breast Cancer Wisconsin Diagnostic or SST-2 for a model
- **THEN** the results contain accuracy, macro-F1, precision, recall and F1 per class, ROC-AUC and PR-AUC for the positive class, and the 2×2 confusion matrix

#### Scenario: Multiclass dataset metrics
- **WHEN** evaluation runs on AG News or Banking77 for a model
- **THEN** the results contain accuracy, macro-F1, precision, recall and F1 for every class, and one-vs-rest macro ROC-AUC and PR-AUC

#### Scenario: Same examples for both models
- **WHEN** metrics are computed for any two of the four models on a dataset
- **THEN** both are computed on the same example identifiers, and the evaluation stops with an error if the identifier sets differ

#### Scenario: Thinking subsample compared on its own items
- **WHEN** Qwen with thinking enabled is reported for a dataset
- **THEN** the other models' metrics shown next to it are computed on the same subsample identifiers
