## REMOVED Requirements

### Requirement: Standard datasets with frozen splits
**Reason**: The tabular datasets no longer use repeated stratified splits over all rows.
**Migration**: Replaced by "Datasets with balanced tabular splits".

### Requirement: Probabilistic prediction interface
**Reason**: It allowed exactly four models and no further Jev request.
**Migration**: Replaced by "Model prediction interface".

## ADDED Requirements

### Requirement: Datasets with balanced tabular splits
The harness SHALL provide German Credit, Breast Cancer Wisconsin Diagnostic, SST-2 and AG News, each with a fixed, ordered class set and, for each example, its raw input, its text serialization and its gold label. SST-2 and AG News SHALL be evaluated on their complete official held-out split, with training and validation data drawn from the official training split. German Credit and Breast Cancer Wisconsin Diagnostic SHALL be undersampled with a fixed seed to equal class counts and split once, with a fixed seed, into training, validation and test parts that each have equal class counts. The balanced identifiers SHALL be frozen in the manifest, with the date, before any model is trained or queried on them; the frozen files of the text datasets SHALL NOT change.

#### Scenario: Official held-out split
- **WHEN** SST-2 or AG News is loaded
- **THEN** its evaluation examples are exactly the official held-out split, and no training or validation example comes from that split

#### Scenario: Balanced parts
- **WHEN** German Credit or Breast Cancer Wisconsin Diagnostic is loaded
- **THEN** its training, validation and test parts are disjoint, each has the same number of rows of the two classes, and two loads give identical identifiers

#### Scenario: Rows left out are recorded
- **WHEN** the balanced subset is drawn
- **THEN** the identifiers of the majority-class rows that were left out are saved, and no stage uses them

#### Scenario: Tabular serialization
- **WHEN** a German Credit or Breast Cancer Wisconsin Diagnostic row is serialized with the original prompt
- **THEN** the serialization contains every feature as a human-readable `name: value` pair, uses decoded category labels rather than codes, writes a missing value as "unknown", and never contains the gold label

#### Scenario: Text serialization
- **WHEN** an SST-2 or AG News example is serialized for Jev
- **THEN** the text is passed as the input with only the documented normalisation applied, and the gold label is not part of it

### Requirement: Model prediction interface
The harness SHALL support these model entries: Jev with its original prompt and Qwen3.6-27B, both zero-shot; Jev with the engineered prompt on the tabular datasets, which uses training labels; and the supervised XGBoost and fine-tuned BERT baselines. For each input, each entry SHALL provide a probability vector over the dataset's full ordered class set that sums to 1 (within 1e-6), the argmax label, and metadata (model version or revision, prompt hash for Jev, and latency and token usage where available). No other model SHALL be trained or queried. New Jev requests SHALL be made only by the prompt-engineering stage, within its budget; every other Jev result comes from saved predictions.

#### Scenario: Full distribution from Jev
- **WHEN** Jev is queried with a Choice question whose options are the dataset classes
- **THEN** the returned vector contains a probability for every class, in the dataset's class order

#### Scenario: Missing Jev response
- **WHEN** an item has no Jev response (for example because its request failed)
- **THEN** it is recorded as an invalid-output event with a uniform distribution rather than silently mapped to a class

#### Scenario: Distribution from the other models
- **WHEN** XGBoost, BERT or Qwen predictions are read for a dataset
- **THEN** each evaluation example has a probability for every class, in the dataset's class order, and the model's version or revision is recorded

#### Scenario: Only prompt engineering sends Jev requests
- **WHEN** every stage other than prompt engineering has run
- **THEN** the Jev ledger total is the same as before they ran

#### Scenario: Two Jev entries on the same rows
- **WHEN** Jev's original and engineered prompts are scored on a tabular dataset
- **THEN** both are scored on the same balanced test identifiers and are stored and reported as separate entries
