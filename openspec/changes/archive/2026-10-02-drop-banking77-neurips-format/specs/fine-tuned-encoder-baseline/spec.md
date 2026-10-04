## MODIFIED Requirements

### Requirement: Fine-tuned encoder on every dataset
The study SHALL fine-tune a pinned revision of a pretrained BERT-base encoder with a classification head on each of the four datasets, and for every repeat of the repeated-split datasets, and SHALL produce a probability vector over the dataset's full ordered class set for exactly the frozen evaluation identifiers used for Jev. Tabular rows SHALL be given to the encoder as the same serialized text Jev receives. The encoder's model identifier and revision SHALL be saved with the predictions.

#### Scenario: Same identifiers as Jev
- **WHEN** encoder predictions are read for a dataset
- **THEN** their identifiers equal the frozen evaluation identifiers, every row has a probability for every class in the dataset's class order, and reading fails if any identifier is missing or extra

#### Scenario: Tabular input
- **WHEN** the encoder is trained on German Credit or Breast Cancer Wisconsin Diagnostic
- **THEN** each row is the serialized text used in the Jev request for that row
