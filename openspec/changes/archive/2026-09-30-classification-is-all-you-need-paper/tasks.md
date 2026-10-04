# Tasks

## 1. Environment and scaffolding

- [x] 1.1 Install `uv`, create a Python 3.11 project (`pyproject.toml`, `uv.lock`) with pinned deps (typesafe-sdk, python-dotenv, xgboost, scikit-learn, shap, lime, openml, datasets, pandas, numpy, matplotlib, pyyaml, pytest) and a `make setup` target; verify `make setup && uv run python -c "import typesafe_sdk, xgboost, shap, lime"` succeeds
- [x] 1.2 Create the `src/ciayn/` package layout, `configs/`, `results/`, `cache/`, `paper/`, `docs/` and extend the existing `.gitignore` (keep `.env`; add cache, results/live raw responses), plus a `python -m ciayn --help` CLI skeleton with stages `smoke|predict|metrics|thresholding|triage|importance|explain|paper` and `--mode {mock,live}`; verify `uv run python -m ciayn --help` lists all stages and `uv run pytest` runs (0 tests OK)
- [x] 1.3 After 3.3, remove the `xgboost` and `datasets` dependencies from `pyproject.toml` and `uv.lock` (Jev is the only model; AG News is dropped); verify `uv sync` succeeds, `uv run python -c "import typesafe_sdk, shap, lime"` works and `grep -riE 'xgboost|^name = "datasets"' pyproject.toml uv.lock` finds nothing

## 2. Datasets

- [x] 2.1 Implement German Credit (OpenML 31), Adult (OpenML 1590) and AG News (HF `fancyzhx/ag_news`) loaders with fixed ordered class lists and stratified seeded splits per design D3, persisting split IDs to `results/splits/*.json`; verify with a pytest that two loads with seed 0 give identical IDs and the expected split sizes
- [x] 2.2 Implement tabular-to-text serialization (`name: value` per line, decoded categories, a "missing fields are unknown" convention) and the Claude-subset selection (300 test / 200 val, stratified); verify with pytests that no serialization contains the gold label and every feature appears exactly once
- [x] 2.3 Restrict the data layer to German Credit (drop the Adult and AG News loaders and the train split) and replace the Claude subset with the stratified, seeded Jev subset of 39 val / 60 test items (design D3); verify with pytests that two loads give identical pool and subset IDs and that the subset has exactly 39 val / 60 test items

## 3. Backends, cache and budget

- [x] 3.1 Implement the `Prediction` type and backend base interface, plus the sqlite response cache keyed on (backend, model_version, question_def, input); verify with a pytest using a counting fake backend that a warm rerun makes zero calls and returns identical outputs, and that probability vectors sum to 1 within 1e-6
- [x] 3.2 Implement the call-budget guard with dry-run counting; verify with a pytest that exceeding the configured budget raises before any call and reports required vs allowed counts
- [x] 3.3 After 3.8 (the mock no longer needs XGBoost), remove the XGBoost backend (`backends/xgb.py`, `xgb` in `config.BACKENDS` and the backend factory, and its tests); verify `grep -ri xgb src tests configs` finds nothing and `uv run pytest` passes
- [x] 3.4 Implement the mock backend (mock-jev and mock-claude profiles: seeded logit noise + temperature over XGBoost probabilities, `meta.mode="mock"`); verify with a pytest that it is deterministic per seed and every output is mock-labelled
- [x] 3.5 Implement the Jev backend in `backends/jev.py` (AsyncTypeSafeClient, concurrency 8, SDK retries disabled, one Choice per example with per-class criteria definitions, pinned model version, full `probabilities` mapped to class order), plus `.env` loading at CLI start-up via python-dotenv (an exported variable wins) and a fail-fast error pointing to `docs/JEV_ACCESS.md` when `TYPESAFE_API_KEY` is missing in live mode; verify with pytests that use a stubbed SDK response to check the mapping, the missing-key error, key loading from a temporary `.env`, and that the key string never appears in cache, log or result files
- [x] 3.6 Remove the Claude backend and every Claude code path (backend module, mock-claude profile, Claude budgets and subset); verify `grep -ri claude src tests configs` finds nothing and `uv run pytest` passes
- [x] 3.7 Write `docs/JEV_ACCESS.md` (request early access at console.typesafe.ai or via OpenRouter, create the key, add `TYPESAFE_API_KEY=...` to the gitignored `.env` (or export it), `make smoke`, the 100-inference cap and its per-stage allocation, estimated full-suite cost) and implement the `smoke` stage (one Choice query, print full distribution + model version); verify the smoke stage prints the missing-key guidance when no key is set, and passes the stubbed-SDK test
- [x] 3.8 Rebuild the mock backend as mock-jev only, derived from gold labels (seeded per-item noise on a gold-class margin + temperature, plus the seeded feature-removal probability function for explainability; no trained model), and update `configs/mock.yaml`; verify with a pytest that it is deterministic per seed and order-independent, and every output is mock-labelled
- [x] 3.9 Implement the global Jev inference cap (design D5): `jev_total: 100` plus per-stage budgets (smoke 1, predict 99, explain 0, others 0) in `configs/budgets.yaml`, and a persistent ledger in the sqlite cache that records every Jev request before it is sent, successful or not, and is checked in preflight; verify with pytests using a counting fake SDK that the cap holds across separate invocations, failed requests are counted, a stage that would exceed 100 stops before any call, and a warm rerun leaves the ledger unchanged

## 4. Prediction stage

- [x] 4.1 Implement the `predict` stage that fills the cache for Jev on the German Credit Jev subset (39 val + 60 test), writing `results/<mode>/german/jev/predictions.parquet` and printing the dry-run report (calls needed, ledger total, cap) first; verify `make predict MODE=mock` completes end-to-end and a second run makes zero calls

## 5. Classification evaluation (metrics and calibration)

- [x] 5.1 Implement the metrics module (accuracy, per-class P/R/F1, macro/micro F1, confusion matrix, ROC-AUC/PR-AUC, missing/invalid Jev outputs counted as errors, 1,000-resample bootstrap CIs); verify with pytests against hand-computed values on a toy set, and check that invalid outputs lower accuracy
- [x] 5.2 Implement calibration (15-bin ECE, Brier, log-loss, reliability diagram); verify with a pytest that a perfectly calibrated synthetic predictor has ECE ≈ 0, and that `python -m ciayn metrics --mode mock` writes the table and the reliability figure for Jev on German Credit

## 6. Cost-sensitive thresholding

- [x] 6.1 Add `configs/costs.yaml` (German Credit 5:1 canonical) and implement the Bayes-optimal rule and cost evaluation; verify with a pytest on a hand-built example
- [x] 6.2 Implement the seeded NumPy GA over class weights (pop 60, 80 gens, tournament, blend crossover, log-space mutation, elitism 2), fitted on val only; verify with pytests that same-seed runs are identical and that on synthetic calibrated binary data the GA threshold is within 0.03 of C_FP/(C_FP+C_FN)
- [x] 6.3 Implement the `thresholding` stage (argmax vs Bayes vs GA test cost + P/R/F1 per rule, GA vs analytic threshold difference, cost-curve figures with marked operating points); verify `python -m ciayn thresholding --mode mock` writes the table and the cost-curve figure

## 7. Confidence triage and HITL

- [x] 7.1 Implement routing over a K-sweep (confidence quantiles, 0–100% coverage) with the simulated human as the only escalation target (configurable accuracy and per-item cost), plus the USD total-cost model with λ; verify with pytests that K below the minimum confidence gives full automation, K above the maximum gives full escalation, and a human oracle at accuracy 1.0 makes escalated items correct
- [x] 7.2 Implement the `triage` stage (val-optimal K reported on test, the Jev risk–coverage curve with AURC and a random-ranking reference, λ and human-cost sensitivity plot, note on Jev-confidence ≡ affine(p_max)); verify `python -m ciayn triage --mode mock` writes the optimal-K table, the risk–coverage figure and the sensitivity figure

## 8. Importance-sampling evaluation

- [x] 8.1 Implement uniform and loss-proportional importance-sampling estimators (Hansen–Hurwitz accuracy, weighted-ratio macro-F1, CLT CIs, ε-mixture proposal with α); verify with a pytest on a synthetic pool that the IS estimator's mean over 5,000 trials is within Monte Carlo error of the truth and q > 0 everywhere
- [x] 8.2 Implement the `importance` stage (pool = the 60-item Jev test subset, 1,000 trials × budgets {5, 10, 20, 30}, bias/variance/RMSE/coverage table, variance ratio, RMSE-vs-budget figure, α sensitivity) using cached predictions only; verify `python -m ciayn importance --mode mock` completes with zero remote calls and writes the table and figures

## 9. Perturbation explainability (mock-only)

- [x] 9.1 Implement KernelSHAP (200 coalitions, empty-coalition background) and LIME (200 samples) over the mock-jev feature-removal probability function for 20 seeded German Credit examples from the Jev test subset, with the explain stage's live Jev budget fixed at 0; verify with pytests that SHAP additivity error is reported, every feature gets an attribution, and running the stage in live mode makes zero Jev requests
- [x] 9.2 Implement the deletion faithfulness test (top-k vs 5× random-k, k ∈ {1, 3, 5}; probability drop + flip rate), Kendall τ / top-3 agreement between KernelSHAP and LIME, and model calls per explanation; verify `python -m ciayn explain --mode mock` writes the faithfulness, agreement and call-cost tables plus example attribution figures, all mock-labelled

## 10. Paper

- [x] 10.1 Implement the `paper` stage generator: booktabs tables, `\newcommand` number macros and PDF figures into `paper/generated/`, plus a `\mockdraft` flag when any consumed artifact is mock; verify with a pytest that mock inputs produce the flag and live-only inputs do not
- [x] 10.2 Write `paper/main.tex` and the section skeleton (Abstract, Introduction, Background, Setup, five results sections (explainability as method + call cost), Limitations, Conclusion, References with `refs.bib`), with a `draftwatermark` switch and `make paper`; verify `make paper MODE=mock` produces a PDF with the "DRAFT — MOCK RESULTS" watermark and no LaTeX errors
- [x] 10.3 Write the Background section: measures on (Σ*, F) and (C, 2^C), the pushforward through the parsing map, the invalid-mass term, and which classical tools need a class measure; verify the section compiles and contains all three formal elements
- [x] 10.4 Write the Introduction, Setup, Limitations (no baseline model, single dataset, small N under the 100-inference cap, mock-only explainability, contamination, option order) and Conclusion prose, referencing only generated macros for numbers; verify `grep` finds no hand-typed result numbers in the results sections and the build succeeds

## 11. Live run and integration (key available in `.env`)

- [x] 11.1 With the key loaded from `.env`, run `make smoke`, pin the reported Jev model version in config, and confirm the response shape matches the backend mapping; verify the smoke output shows a full probability distribution
- [x] 11.2 Run `make predict MODE=live` for Jev (dry-run budget report first; at most 99 calls), then all experiment stages in live mode (explain makes no Jev calls); verify all live `results/live/**` artifacts exist with `mode=live`, the Jev ledger total is ≤ 100, and a second full run makes zero remote calls and leaves the ledger unchanged
- [x] 11.3 Fill the results-section prose from the live numbers (via macros), then run `make paper MODE=live`; verify the PDF builds without the mock watermark, states the Jev version, run date, sample sizes, total Jev inferences used (≤ 100) and cost/latency, has no mock-derived explainability numbers, and runs to about 6–8 pages before references
