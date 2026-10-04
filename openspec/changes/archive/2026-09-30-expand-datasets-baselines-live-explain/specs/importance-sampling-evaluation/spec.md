# Spec Delta

## MODIFIED Requirements

### Requirement: Label-budgeted estimation
Given a fixed evaluation pool with cached predictions and a label budget n smaller than the pool, the experiment SHALL estimate the model's accuracy on the pool by sampling n items to label, for both (a) uniform random sampling and (b) confidence-guided importance sampling with a proposal q that increases with the model's predicted uncertainty (1 − max class probability) and is mixed with the uniform distribution so that q > 0 for every item. The estimate SHALL use an inverse-probability-weighted estimator that is unbiased for the pool accuracy under the sampling design. Accuracy SHALL be the only estimated quantity.

#### Scenario: Unbiased weighting
- **WHEN** importance sampling selects item i with probability q_i
- **THEN** its contribution to the estimate is weighted by p_i / q_i, where p is the uniform target distribution over the pool

#### Scenario: Positive sampling probability
- **WHEN** the proposal is built for a pool, including items the model predicts with probability 1
- **THEN** every item has q_i > 0 and the probabilities sum to 1

### Requirement: Estimator quality comparison
The experiment SHALL repeat estimation over at least 500 seeded trials per budget for at least four budgets on each dataset's pool, and SHALL report, per sampler, bias, variance, RMSE against the full-pool accuracy, and empirical 95% interval coverage. The full-pool accuracy SHALL be used only as the ground truth for this comparison.

#### Scenario: Variance reduction report
- **WHEN** trials complete for a budget on a dataset
- **THEN** a table reports bias, variance, RMSE and interval coverage for uniform and importance sampling, and the variance ratio between them

### Requirement: Budget–error visualization
The experiment SHALL produce, for each dataset, a figure of RMSE versus label budget for each sampler.

#### Scenario: Figure output
- **WHEN** the experiment finishes on a dataset
- **THEN** an RMSE-vs-budget figure with one line per sampler is written to the results directory

## ADDED Requirements

### Requirement: Labels hidden from the sampler
The sampling procedure SHALL receive only the model's predictions for the pool. A label SHALL be revealed only for an item the procedure has selected, and the proposal distribution SHALL NOT depend on any label.

#### Scenario: Only sampled labels are revealed
- **WHEN** a trial draws n items from the pool
- **THEN** labels are requested only for the drawn items, and no other label in the pool is read by the sampling or estimation procedure

#### Scenario: Proposal independent of labels
- **WHEN** the pool's labels are permuted while predictions are unchanged
- **THEN** the proposal distribution is identical
