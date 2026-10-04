# Proposal

## Why

TypeSafe's Jev ("System One" model) returns a calibrated probability for **every** option of a predefined class set instead of generating tokens. This moves the output space from a measure over token strings (Σ*) to a measure on a finite class space (C, 2^C). That shift makes decades of classical classification tooling usable again, which token-generating LLMs make awkward or impossible. We want a short paper, *"Classification Is All You Need"*, that makes this argument formally and backs it with reproducible experiments on a standard data-science dataset.

## What Changes

- New Python experiment harness (Python ≥ 3.10, isolated env) with:
  - a loader and fixed splits for **German Credit** (binary, canonical 5:1 cost matrix). Tabular rows are serialized to text for Jev.
  - a single model backend, **Jev**, called through `typesafe-sdk` `Choice` questions and returning a probability vector over classes. There is no classical baseline and no LLM comparison model.
  - a hard cap of **at most 100 Jev inferences for the whole live run** (smoke test included), enforced before any call is made.
  - a disk cache for every model response, so reruns are free and deterministic. A mock mode (a synthetic, clearly labelled "mock-jev") lets the whole pipeline run offline (tests, development) without remote calls.
- Five experiment modules, one per point of the thesis:
  1. **Metrics**: precision, recall, F1 (per class, macro, micro), ROC/PR, plus calibration (ECE, Brier, reliability diagrams) to test Jev's calibration claim.
  2. **Cost-sensitive thresholding**: a genetic algorithm that searches decision thresholds under asymmetric misclassification costs, checked against the analytic Bayes-optimal threshold.
  3. **Confidence triage**: selective classification where items with confidence < K go to a simulated human-in-the-loop (HITL) reviewer. Outputs are risk–coverage and total-cost curves as functions of K.
  4. **Importance-sampling evaluation**: estimate accuracy and F1 from a small labelling budget, drawing samples with a confidence-derived proposal, and compare estimator variance against uniform sampling.
  5. **Perturbation-based explainability**: KernelSHAP and LIME over a masking-based probability function with deletion faithfulness tests, implemented and validated on mock-jev only. Live Jev attributions do not fit the 100-inference cap, so the paper presents the method and its call cost.
- A guide explaining how to get Jev access and configure `TYPESAFE_API_KEY` in a gitignored `.env`.
- A short arXiv-style LaTeX paper (~6–8 pages) whose tables and figures are generated from the experiment outputs.

## Capabilities

### New Capabilities
- `experiment-harness`: environment, dataset loading/splitting/serialization, the Jev backend, the Jev inference cap, response caching, mock mode, and the Jev access guide.
- `classification-evaluation`: classical metrics and calibration analysis over per-class probability outputs.
- `cost-sensitive-thresholding`: GA-based decision-threshold optimization under asymmetric cost matrices, with a Bayes-optimal reference.
- `confidence-triage`: confidence-threshold routing to a simulated human, with cost and risk–coverage analysis.
- `importance-sampling-evaluation`: label-efficient performance estimation with importance-weighted estimators and variance comparison.
- `perturbation-explainability`: SHAP/LIME attributions over a class-probability function and faithfulness metrics, run on mock-jev.
- `paper-manuscript`: the LaTeX paper, its build, and automatic inclusion of generated figures and tables.

### Modified Capabilities
- None (greenfield project).

## Impact

- New code under `src/`, experiment configs, `results/` (cached responses, metrics, figures) and `paper/`. Nothing existing is affected.
- New dependencies: `typesafe-sdk`, `python-dotenv`, `scikit-learn`, `shap`, `lime`, `openml` loader, `matplotlib`, `numpy`, `pandas`. Python ≥ 3.10 is required (the system Python is 3.9).
- External services:
  - **TypeSafe Jev API** only: the user has a key, stored as `TYPESAFE_API_KEY` in a gitignored `.env`. The whole live run makes at most 100 Jev calls; cost is negligible at $0.042/MTok input.
- Compute: a 2-CPU / 3 GB machine; no model is trained locally.
