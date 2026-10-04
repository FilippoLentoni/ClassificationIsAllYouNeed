## MODIFIED Requirements

### Requirement: Probabilistic prediction interface
The harness SHALL support exactly four models: Jev and Qwen3.6-27B, both queried zero-shot, and the supervised XGBoost and fine-tuned BERT baselines. For each input, each model SHALL provide a probability vector over the dataset's full ordered class set that sums to 1 (within 1e-6), the argmax label, and metadata (model version or revision, and latency and token usage where available). No other model SHALL be trained or queried. The only new Jev requests SHALL be at most 1,000 predictions on seeded validation items of AG News and Banking77 for the escalation threshold, made under their own stage budget and counted in the ledger; every other Jev result comes from the saved predictions.

#### Scenario: Full distribution from Jev
- **WHEN** Jev is queried with a Choice question whose options are the dataset classes
- **THEN** the returned vector contains a probability for every class, in the dataset's class order

#### Scenario: Missing Jev response
- **WHEN** an item has no Jev response (for example because its request failed)
- **THEN** it is recorded as an invalid-output event with a uniform distribution rather than silently mapped to a class

#### Scenario: Distribution from the baseline
- **WHEN** baseline predictions are read for a dataset
- **THEN** each evaluation example has a probability for every class, in the dataset's class order

#### Scenario: Distribution from BERT and Qwen
- **WHEN** BERT or Qwen predictions are read for a dataset
- **THEN** each row has a probability for every class in the dataset's class order, and the model's revision is recorded

#### Scenario: Jev ledger unchanged
- **WHEN** the BERT, Qwen, comparison and escalation stages have all run
- **THEN** the Jev ledger total is the same as before they ran

#### Scenario: Escalation validation requests bounded
- **WHEN** the escalation validation predictions are requested
- **THEN** at most 1,000 Jev requests are made, all on validation items of AG News and Banking77, and none on evaluation items
