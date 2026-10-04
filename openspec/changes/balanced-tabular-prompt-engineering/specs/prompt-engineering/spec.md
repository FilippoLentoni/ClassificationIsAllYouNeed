## Purpose

Tests whether Jev's decisions on the tabular datasets can be improved by changing only its prompt, using training data alone, with every step recorded so that the improvement can be audited.

## ADDED Requirements

### Requirement: Training-data-only prompt development
Prompt engineering for Jev SHALL treat the prompt (instructions, option descriptions and input serialization) as the parameter being fitted. Every prompt version SHALL be scored on rows of the balanced training part only. No validation or test row SHALL be sent to Jev with a candidate prompt, and no validation or test result SHALL be inspected, until the candidate prompts are fixed.

#### Scenario: Only training rows are used during development
- **WHEN** a candidate prompt is scored during a development round
- **THEN** every Jev request of that round is for a training-part row, and the stage refuses a request for any other row

### Requirement: Error analysis drives each round
Each round SHALL start from the results of the previous prompt on the training rows: the confusion counts, the confidence of the errors and the features or wording associated with them. The round's record SHALL state the gap that was found, the change made to the prompt in response, and the training results before and after. A change SHALL NOT be made without a recorded observation that motivates it.

#### Scenario: Round recorded
- **WHEN** a development round completes
- **THEN** a saved record contains the prompt text, the observation that motivated the change, the change itself and the training accuracy, balanced accuracy and confusion counts before and after

#### Scenario: First round is the diagnosis
- **WHEN** development starts for a dataset
- **THEN** round zero records the original prompt's results on the training rows and the gaps found, and changes nothing

### Requirement: Validation choice and single test run
The prompt to report SHALL be chosen among the recorded versions by accuracy on the validation part, which is used only for that choice. The chosen prompt SHALL be frozen, with its hash, before any test row is sent with it, and SHALL be run on the test part exactly once.

#### Scenario: Frozen before the test run
- **WHEN** the test part is predicted with the engineered prompt
- **THEN** the prompt's hash equals the hash recorded when it was chosen on validation, and no later prompt version exists

### Requirement: Both prompts reported
Results SHALL be reported for Jev with the original prompt and for Jev with the engineered prompt on the same test rows. The engineered prompt SHALL be reported as using the training and validation labels it was developed and chosen on, and SHALL NOT be called zero-shot. If no version beats the original prompt on validation, the original SHALL be reported as the outcome, with the rounds that were tried.

#### Scenario: No improvement
- **WHEN** no engineered version has higher validation accuracy than the original prompt
- **THEN** the results say so, the original prompt is reported, and every round that was tried remains in the record

### Requirement: Prompt-engineering budget
The stage SHALL have its own Jev request budget of at most 7,000 requests, counted in the ledger and checked before any request is sent, within the study's cap.

#### Scenario: Budget enforced
- **WHEN** a round would exceed the stage budget or the study's cap
- **THEN** no request is sent and the required and allowed counts are reported
