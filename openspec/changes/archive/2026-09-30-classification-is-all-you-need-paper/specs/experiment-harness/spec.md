# Spec Delta

## Purpose

Provides the reproducible foundation for all experiments: a dataset with fixed splits, a probabilistic interface to Jev (the only model), a hard cap on Jev inferences, response caching, an offline mock mode, and instructions for obtaining Jev access.

## ADDED Requirements

### Requirement: Isolated reproducible environment
The project SHALL provide an isolated Python environment (Python ≥ 3.10) with pinned dependencies that can be created with a single documented command, independent of the system Python.

#### Scenario: Fresh environment setup
- **WHEN** a user runs the documented setup command on a machine with only the system Python 3.9
- **THEN** an isolated Python ≥ 3.10 environment is created with all pinned dependencies installed and the test suite runs

### Requirement: Standard dataset with fixed splits
The harness SHALL provide German Credit with deterministic, seeded validation/test pools and a stratified, seeded Jev subset of 39 validation and 60 test items, exposing for each example its raw features, its text serialization and its gold label from a fixed, ordered class set. No training split SHALL be required, since no model is trained.

#### Scenario: Deterministic splits
- **WHEN** the dataset is loaded twice with the same seed
- **THEN** the validation and test pool identifiers and the Jev subset identifiers are identical across both loads

#### Scenario: Jev subset size
- **WHEN** the Jev subset is drawn
- **THEN** it contains exactly 39 validation and 60 test items, stratified by class

#### Scenario: Tabular serialization
- **WHEN** a German Credit row is serialized for Jev
- **THEN** the serialization contains every feature as a human-readable `name: value` pair, uses decoded category labels rather than codes, and never contains the gold label

### Requirement: Probabilistic prediction interface
Jev SHALL be the only model backend; the harness SHALL NOT train or query any other model (no classical baseline, no other LLM). The Jev backend and its mock SHALL return, for each input, a probability vector over the dataset's full ordered class set that sums to 1 (within 1e-6), the argmax label, and metadata (latency, token usage where available, model version).

#### Scenario: Full distribution from Jev
- **WHEN** Jev is queried with a Choice question whose options are the dataset classes
- **THEN** the returned vector contains a probability for every class, in the dataset's class order

#### Scenario: Missing Jev response
- **WHEN** an item has no Jev response (for example because its request failed)
- **THEN** it is recorded as an invalid-output event with a uniform distribution rather than silently mapped to a class

### Requirement: Response caching and deterministic reruns
All remote model responses SHALL be persisted to an on-disk cache keyed by backend, model version, prompt/question definition and input, so that re-running any experiment with a warm cache makes zero remote calls and produces identical results.

#### Scenario: Warm-cache rerun
- **WHEN** an experiment is re-run after a complete prior run
- **THEN** no remote API calls are made and all output metrics and figures are byte-identical

### Requirement: Offline mock mode
The harness SHALL support a mock mode, selected by configuration, in which the Jev backend is replaced by a clearly labelled, seeded synthetic predictor ("mock-jev") derived from the gold labels, without any trained model, so that the full pipeline runs end-to-end without API keys. Outputs produced in mock mode SHALL be marked as mock in every result file and figure.

#### Scenario: Deterministic mock
- **WHEN** mock-jev predicts the same items twice with the same seed, in any order
- **THEN** the probability vectors are identical

#### Scenario: Running without a key
- **WHEN** the pipeline runs with no `TYPESAFE_API_KEY` set and mock mode enabled
- **THEN** all experiments complete and every produced artifact is labelled as mock

#### Scenario: Missing key without mock mode
- **WHEN** a Jev experiment runs with no `TYPESAFE_API_KEY` set and mock mode disabled
- **THEN** the run fails immediately with a message pointing to the Jev access guide

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
Every experiment SHALL declare the maximum number of Jev calls it will make and SHALL report the calls it would make (dry-run) before spending, refusing to exceed the configured budget.

#### Scenario: Budget exceeded
- **WHEN** a configured stage would require more Jev calls than its budget
- **THEN** the run stops before making calls and reports the required and allowed counts

### Requirement: Jev inference cap
The harness SHALL make at most 100 Jev inferences in total across the whole live run, including the smoke test, all stages and all separate invocations. Every Jev request SHALL be recorded in a persistent ledger before it is sent, and SHALL count against the cap whether or not it succeeds. SDK-level automatic retries SHALL be disabled so that each request is counted exactly once.

#### Scenario: Cap enforced before any call
- **WHEN** a stage's dry-run count plus the ledger total would exceed 100
- **THEN** the stage stops before making any Jev call and reports the calls required, the ledger total and the cap

#### Scenario: Cap holds across invocations
- **WHEN** several live CLI invocations run one after another, including after a crash
- **THEN** the ledger total never exceeds 100

#### Scenario: Warm rerun costs nothing
- **WHEN** the full live pipeline is re-run with a warm cache
- **THEN** no Jev requests are made and the ledger total is unchanged
