# experiment-harness Specification

## Purpose

Provides the reproducible foundation for all experiments: four public datasets with frozen evaluation identifiers and documentation, a probabilistic interface to the four models of the study (Jev and Qwen3.6-27B zero-shot, XGBoost and BERT supervised), a hard cap on Jev inferences, response caching, a request and response log, persisted evaluation artifacts, an offline mock mode, and instructions for obtaining Jev access.

## Requirements

### Requirement: Isolated reproducible environment
The project SHALL provide an isolated Python environment (Python ≥ 3.10) with pinned dependencies that can be created with a single documented command, independent of the system Python.

#### Scenario: Fresh environment setup
- **WHEN** a user runs the documented setup command on a machine with only the system Python 3.9
- **THEN** an isolated Python ≥ 3.10 environment is created with all pinned dependencies installed and the test suite runs

### Requirement: Probabilistic prediction interface
The harness SHALL support exactly four models: Jev and Qwen3.6-27B, both queried zero-shot, and the supervised XGBoost and fine-tuned BERT baselines. For each input, each model SHALL provide a probability vector over the dataset's full ordered class set that sums to 1 (within 1e-6), the argmax label, and metadata (model version or revision, and latency and token usage where available). No other model SHALL be trained or queried. No further Jev request SHALL be made: every Jev result comes from the saved predictions, including the 500 seeded AG News validation items predicted for the escalation threshold.

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
- **THEN** no Jev request is made: the saved predictions for the 500 AG News validation items are read, none of which is an evaluation item

### Requirement: Response caching and deterministic reruns
All remote model responses SHALL be persisted to an on-disk cache keyed by backend, model version, prompt/question definition and input, so that re-running any experiment with a warm cache makes zero remote calls and produces identical results.

#### Scenario: Warm-cache rerun
- **WHEN** an experiment is re-run after a complete prior run
- **THEN** no remote API calls are made and all output metrics and figures are byte-identical

### Requirement: Offline mock mode
The harness SHALL support a mock mode, selected by configuration, in which Jev and the supervised baseline are replaced by clearly labelled, seeded synthetic predictors, without any trained model, so that the full pipeline runs end-to-end for all four datasets without an API key and without AWS access. Outputs produced in mock mode SHALL be marked as mock in every result file and figure.

#### Scenario: Deterministic mock
- **WHEN** a mock predictor predicts the same items twice with the same seed, in any order
- **THEN** the probability vectors are identical

#### Scenario: Running without a key
- **WHEN** the pipeline runs in mock mode with no `TYPESAFE_API_KEY` set and no AWS credentials
- **THEN** all experiments complete for all four datasets and every produced artifact is labelled as mock

#### Scenario: Missing key without mock mode
- **WHEN** a live stage needs to make a Jev request and no `TYPESAFE_API_KEY` is set
- **THEN** the run fails before any request with a message pointing to the Jev access guide

### Requirement: API key handling
The harness SHALL read `TYPESAFE_API_KEY` from the environment, loading a gitignored `.env` file at the project root when the variable is not already set. The key SHALL NOT be written to any tracked file, log, cache entry, cache key, or result artifact.

#### Scenario: Key loaded from .env
- **WHEN** `TYPESAFE_API_KEY` is not exported and `.env` defines it
- **THEN** live-mode Jev calls authenticate using the value from `.env`

#### Scenario: Exported variable takes precedence
- **WHEN** `TYPESAFE_API_KEY` is exported and `.env` also defines it
- **THEN** the exported value is used

#### Scenario: Key never persisted
- **WHEN** a live run completes
- **THEN** the key string does not appear in any file under `cache/`, `results/` or the run logs

### Requirement: Jev access guide
The project SHALL include a guide explaining how to obtain Jev early access, create an API key, set `TYPESAFE_API_KEY` in the gitignored `.env` or the environment, verify connectivity with a single smoke-test command, and estimate the API cost of the full experiment suite.

#### Scenario: Smoke test
- **WHEN** a user follows the guide and runs the smoke-test command with a valid key
- **THEN** a single Jev Choice query is made and its full class distribution is printed

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

### Requirement: Standard datasets with frozen splits
The harness SHALL provide German Credit, Breast Cancer Wisconsin Diagnostic, SST-2 and AG News, each with a fixed, ordered class set and, for each example, its raw input, its text serialization and its gold label. SST-2, AG News and Banking77 SHALL be evaluated on their complete official held-out split (the labelled validation split for SST-2; the test split for AG News and Banking77), and any training or validation data for them SHALL be drawn from the official training split. German Credit and Breast Cancer Wisconsin Diagnostic SHALL use repeated, seeded, stratified train/validation/test splits over all of their rows.

#### Scenario: Official held-out split
- **WHEN** SST-2 or AG News is loaded
- **THEN** its evaluation examples are exactly the official held-out split, and no training or validation example comes from that split

#### Scenario: Deterministic repeated splits
- **WHEN** German Credit or Breast Cancer Wisconsin Diagnostic is loaded twice with the same seed
- **THEN** the train, validation and test identifiers of every repeat are identical across both loads, are disjoint within a repeat, and preserve the class proportions

#### Scenario: Tabular serialization
- **WHEN** a German Credit or Breast Cancer Wisconsin Diagnostic row is serialized for Jev
- **THEN** the serialization contains every feature as a human-readable `name: value` pair, uses decoded category labels rather than codes, writes a missing value as "unknown", and never contains the gold label

#### Scenario: Text serialization
- **WHEN** an SST-2 or AG News example is serialized for Jev
- **THEN** the text is passed as the input with only the documented normalisation applied, and the gold label is not part of it

### Requirement: Frozen evaluation identifiers
The harness SHALL write the training, validation and evaluation identifiers of every dataset, including every split repeat, to a manifest with a content hash before any of them is used for baseline training, hyperparameter selection, threshold selection or prompt-template checking. Every later stage SHALL verify the manifest hash and refuse to run if it differs. Prompt templates SHALL be checked only on training-split examples.

#### Scenario: Manifest changed after freezing
- **WHEN** a stage runs and the identifiers on disk do not match the recorded manifest hash
- **THEN** the stage stops before doing any work and reports the mismatch

#### Scenario: Template check stays off the evaluation set
- **WHEN** a prompt template is tried on examples before the evaluation run
- **THEN** every example used comes from the training split and none is an evaluation or validation identifier

#### Scenario: A dataset is removed after freezing
- **WHEN** a dataset is removed from the study
- **THEN** the split file and manifest hash of every remaining dataset are byte-identical to those written at the freeze, only the removed dataset's entry leaves the manifest, and the removal is recorded with its date and the unchanged hashes

### Requirement: Dataset documentation
The harness SHALL produce, for each dataset, a record of its source and version or revision, its class definitions and class order, its positive class where binary, its split sizes and number of repeats, its preprocessing and serialization rules including missing-value handling, and its seeds.

#### Scenario: Dataset record generated
- **WHEN** the datasets are frozen
- **THEN** a documentation file exists for each of the four datasets containing every listed item, generated from the same configuration the experiments use

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

### Requirement: Accounting for a removed dataset
Work already paid for on a dataset that is later removed from the study SHALL stay in every total. The Jev requests made on Banking77 SHALL remain in the ledger, the request log and the reported Jev total, and its SageMaker executions SHALL remain in the reported AWS total. The number of Jev requests and the cost attributable to the removed dataset SHALL be saved and reported separately. Removing the dataset SHALL make no Jev request and start no SageMaker job.

#### Scenario: Totals unchanged by the removal
- **WHEN** usage is computed after Banking77 has been removed
- **THEN** the Jev ledger total and the recorded AWS cost equal their values before the removal

#### Scenario: Removed share reported
- **WHEN** the paper and the report state the Jev and AWS totals
- **THEN** they also state how many Jev requests and how much Jev and AWS cost belong to the removed dataset

#### Scenario: No results remain for the removed dataset
- **WHEN** the stages and the paper build run after the removal
- **THEN** no table, figure, macro or results file refers to Banking77 except the accounting of what was spent on it
