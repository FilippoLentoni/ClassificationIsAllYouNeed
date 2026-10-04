# Spec Delta

## MODIFIED Requirements

### Requirement: Probabilistic prediction interface
The harness SHALL support exactly two models: Jev, queried zero-shot, and the supervised baseline. For each input, each model SHALL provide a probability vector over the dataset's full ordered class set that sums to 1 (within 1e-6), the argmax label, and metadata (model version, and for Jev latency and token usage where available). No other model SHALL be trained or queried.

#### Scenario: Full distribution from Jev
- **WHEN** Jev is queried with a Choice question whose options are the dataset classes
- **THEN** the returned vector contains a probability for every class, in the dataset's class order

#### Scenario: Missing Jev response
- **WHEN** an item has no Jev response (for example because its request failed)
- **THEN** it is recorded as an invalid-output event with a uniform distribution rather than silently mapped to a class

#### Scenario: Distribution from the baseline
- **WHEN** baseline predictions are read for a dataset
- **THEN** each evaluation example has a probability for every class, in the dataset's class order

### Requirement: Offline mock mode
The harness SHALL support a mock mode, selected by configuration, in which Jev and the supervised baseline are replaced by clearly labelled, seeded synthetic predictors, without any trained model, so that the full pipeline runs end-to-end for all five datasets without an API key and without AWS access. Outputs produced in mock mode SHALL be marked as mock in every result file and figure.

#### Scenario: Deterministic mock
- **WHEN** a mock predictor predicts the same items twice with the same seed, in any order
- **THEN** the probability vectors are identical

#### Scenario: Running without a key
- **WHEN** the pipeline runs in mock mode with no `TYPESAFE_API_KEY` set and no AWS credentials
- **THEN** all experiments complete for all five datasets and every produced artifact is labelled as mock

#### Scenario: Missing key without mock mode
- **WHEN** a live stage needs to make a Jev request and no `TYPESAFE_API_KEY` is set
- **THEN** the run fails before any request with a message pointing to the Jev access guide

### Requirement: Budgeted remote usage
Every stage that can make remote calls (Jev requests or SageMaker jobs) SHALL declare the maximum it will make and SHALL, on a dry run, report what it would make and stop without making any. It SHALL refuse to exceed its configured budget.

#### Scenario: Budget exceeded
- **WHEN** a configured stage would require more Jev calls than its budget
- **THEN** the run stops before making calls and reports the required and allowed counts

#### Scenario: Dry run of a remote stage
- **WHEN** a stage that can make remote calls runs as a dry run
- **THEN** it reports the Jev requests or SageMaker jobs it would make, makes none, and leaves the ledger unchanged

### Requirement: Jev inference cap
The harness SHALL make at most 60,000 Jev inferences in total across the whole study, counting every stage, every separate invocation and the 100 inferences made before this limit was raised. Every Jev request SHALL be recorded in a persistent ledger before it is sent, and SHALL count against the cap whether or not it succeeds. SDK-level automatic retries SHALL be disabled so that each request is counted exactly once.

#### Scenario: Cap enforced before any call
- **WHEN** a stage's dry-run count plus the ledger total would exceed 60,000
- **THEN** the stage stops before making any Jev call and reports the calls required, the ledger total and the cap

#### Scenario: Cap holds across invocations
- **WHEN** several live CLI invocations run one after another, including after a crash
- **THEN** the ledger total never exceeds 60,000

#### Scenario: Earlier inferences count
- **WHEN** the ledger already holds the 100 inferences made under the earlier limit
- **THEN** they are included in the total compared against the cap

#### Scenario: Warm rerun costs nothing
- **WHEN** the full live pipeline is re-run with a warm cache
- **THEN** no Jev requests are made and the ledger total is unchanged

## REMOVED Requirements

### Requirement: Standard dataset with fixed splits
**Reason**: The study moves from one dataset with a 39/60 Jev subset to five datasets evaluated on official held-out splits or repeated stratified splits; the fixed subset no longer exists.
**Migration**: Replaced by "Standard datasets with frozen splits". The 99 cached German Credit responses are reused through the response cache; the old split file and subset are deleted.

## ADDED Requirements

### Requirement: Standard datasets with frozen splits
The harness SHALL provide German Credit, Breast Cancer Wisconsin Diagnostic, SST-2, AG News and Banking77, each with a fixed, ordered class set and, for each example, its raw input, its text serialization and its gold label. SST-2, AG News and Banking77 SHALL be evaluated on their complete official held-out split (the labelled validation split for SST-2; the test split for AG News and Banking77), and any training or validation data for them SHALL be drawn from the official training split. German Credit and Breast Cancer Wisconsin Diagnostic SHALL use repeated, seeded, stratified train/validation/test splits over all of their rows.

#### Scenario: Official held-out split
- **WHEN** SST-2, AG News or Banking77 is loaded
- **THEN** its evaluation examples are exactly the official held-out split, and no training or validation example comes from that split

#### Scenario: Deterministic repeated splits
- **WHEN** German Credit or Breast Cancer Wisconsin Diagnostic is loaded twice with the same seed
- **THEN** the train, validation and test identifiers of every repeat are identical across both loads, are disjoint within a repeat, and preserve the class proportions

#### Scenario: Tabular serialization
- **WHEN** a German Credit or Breast Cancer Wisconsin Diagnostic row is serialized for Jev
- **THEN** the serialization contains every feature as a human-readable `name: value` pair, uses decoded category labels rather than codes, writes a missing value as "unknown", and never contains the gold label

#### Scenario: Text serialization
- **WHEN** an SST-2, AG News or Banking77 example is serialized for Jev
- **THEN** the text is passed as the input with only the documented normalisation applied, and the gold label is not part of it

### Requirement: Frozen evaluation identifiers
The harness SHALL write the training, validation and evaluation identifiers of every dataset, including every split repeat, to a manifest with a content hash before any of them is used for baseline training, hyperparameter selection, threshold selection or prompt-template checking. Every later stage SHALL verify the manifest hash and refuse to run if it differs. Prompt templates SHALL be checked only on training-split examples.

#### Scenario: Manifest changed after freezing
- **WHEN** a stage runs and the identifiers on disk do not match the recorded manifest hash
- **THEN** the stage stops before doing any work and reports the mismatch

#### Scenario: Template check stays off the evaluation set
- **WHEN** a prompt template is tried on examples before the evaluation run
- **THEN** every example used comes from the training split and none is an evaluation or validation identifier

### Requirement: Dataset documentation
The harness SHALL produce, for each dataset, a record of its source and version or revision, its class definitions and class order, its positive class where binary, its split sizes and number of repeats, its preprocessing and serialization rules including missing-value handling, and its seeds.

#### Scenario: Dataset record generated
- **WHEN** the datasets are frozen
- **THEN** a documentation file exists for each of the five datasets containing every listed item, generated from the same configuration the experiments use

### Requirement: Request and response logging
Every live Jev request SHALL be recorded together with its response: the request content as sent, the response content as received, the SDK version, the requested and reported model versions, a timestamp and the latency. The record SHALL NOT contain the API key or any authentication header. Records SHALL NOT be created, back-filled or edited for exchanges that were not logged when they happened.

#### Scenario: Exchange recorded
- **WHEN** a live Jev request completes
- **THEN** one record holds the request as sent and the response as received, with the SDK and model versions

#### Scenario: Key excluded
- **WHEN** the log is inspected after a live run
- **THEN** neither the API key nor an authentication header appears in it

#### Scenario: No back-fill for earlier calls
- **WHEN** a request made before logging existed is regenerated from code
- **THEN** it is labelled as regenerated, is reported together with whether its hash matches the stored cache key, and is not added to the exchange log

### Requirement: Persisted evaluation artifacts
Predictions for every (dataset, model) pair, the payload examples, and all generated tables and figures SHALL be stored in version-tracked files. The response cache, the ledger and the exchange log SHALL be copied to storage outside the working copy after every live stage.

#### Scenario: Rebuild from tracked files
- **WHEN** the metrics, thresholding and importance-sampling stages and the paper build run from a fresh checkout that has no response cache
- **THEN** they complete from the tracked predictions and results with zero Jev requests and zero SageMaker jobs

#### Scenario: Backup after a live stage
- **WHEN** a live stage finishes
- **THEN** the response cache, ledger and exchange log have been copied to the backup location, and the copy's location and time are reported
