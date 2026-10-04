# Design

## Context

- The project is greenfield: the only contents are `openspec/` and an empty `prompt.md`.
- The host is small (2 CPUs, 3 GB RAM). The system Python is 3.9, but `typesafe-sdk` needs 3.10 or later. `pdflatex` and `latexmk` are installed.
- **Jev API, as documented publicly (Sept 2026)**
  - Install with `pip install typesafe-sdk`; the client is `TypeSafeClient()` / `AsyncTypeSafeClient()`. It reads `TYPESAFE_API_KEY` and defaults to `jev-latest`. Versioned names such as `jev-1.13.0` also exist.
  - Call it as `client.system_one(state, {name: Choice(instructions=..., criteria={label: description})})`. The response gives `choices[name].choice`, `.confidence` and `.probabilities`, which covers **every** label.
  - Limits: 255 options per Choice, a 32k-token state, and 2 retries by default.
  - The endpoint is `POST https://api.typesafe.ai/v1/systemone`. Jev is also reachable through OpenRouter.
  - Pricing is $0.042/MTok input; output is free.
- The user **has a Jev API key**. It is stored as `TYPESAFE_API_KEY` in a gitignored `.env` at the project root (file mode 600). The key never appears in specs, configs, logs, caches or results. Mock mode stays for tests and offline development.
- **Jev is the only model.** There is no classical baseline (no XGBoost, no local training) and no LLM comparison model (no Claude).
- **Hard cap: at most 100 Jev inferences for the whole live run**, smoke test included.

## Goals / Non-Goals

**Goals:**
- One data/prediction layer, with the five experiments running as offline consumers of cached predictions. Only `smoke` and `predict` make remote calls, and together they never exceed 100 Jev inferences.
- Every reported number can be traced to a cached response and regenerated with one command.
- The pipeline works fully in mock mode (tests, offline work), and `--mode live` switches to real Jev calls using the key from `.env`.

**Non-Goals:**
- Fine-tuning or prompt-optimizing Jev beyond a single, documented zero-shot question design per dataset.
- Comparing Jev against baselines (classical models or other LLMs), or chasing accuracy. The paper is about which *tooling* a class distribution unlocks, not a leaderboard.
- Live Jev explainability. KernelSHAP/LIME need about 200 calls per example, far beyond the 100-inference cap; they run on mock-jev only.
- Generic benchmark infrastructure for other models or datasets.

## Decisions

### D1. Environment: `uv`-managed Python 3.11 venv
`uv` installs Python 3.11 without root and pins dependencies through `uv.lock`. The setup command is `make setup`.
- *Alternatives:* pyenv (slow to build on a 2-CPU box) and conda (heavy and slower to resolve).

### D2. Layout
```
src/ciayn/
  data/        # loaders, splits, serialization
  backends/    # jev.py, mock.py, cache.py, budget.py, base.py
  experiments/ # metrics.py, thresholding.py, triage.py, importance.py, explain.py
  paper/       # table/figure generators
configs/       # one YAML per experiment + datasets.yaml + costs.yaml
results/       # {mock|live}/<dataset>/<backend>/... metrics JSON, figures
cache/         # sqlite response cache (gitignored)
paper/         # main.tex, sections/*.tex, generated/ (auto)
docs/JEV_ACCESS.md
Makefile       # setup, smoke, predict, experiments, paper, all
```
A CLI (`python -m ciayn <stage> --mode {mock,live}`) drives the stages. `results/mock` and `results/live` are kept apart so mock numbers can never leak into the live paper.

### D3. Datasets and splits

| Dataset | Source | Classes | Val pool | Test pool | Jev subset (val / test) |
|---|---|---|---|---|---|
| German Credit | OpenML `credit-g` (id 31) | good / bad | 200 | 300 | 39 / 60 |

- German Credit is the only dataset. Adult Income and AG News are dropped: the 100-inference cap cannot give meaningful samples on more than one dataset.
- No train split is needed, because nothing is trained. Val and test pools are stratified with seed 0, and their IDs are saved to `results/splits/german.json`.
- The **Jev subset** is a stratified, seeded (seed 0) draw of 39 val and 60 test items from those pools. Every experiment uses the Jev subset: validation fitting on the 39 val items, reporting on the 60 test items.
- Tabular rows are serialized one feature per line as `name: value`, with decoded categorical labels.

### D4. Backend interface
Each backend implements `predict(examples) -> list[Prediction]`. A `Prediction` holds:
- `probs` (ndarray in class order)
- `label`
- `valid: bool`
- `meta` (latency, tokens or cost, model version, `mode`)

The `cache.py` wrapper keys the sqlite cache on `sha256(backend, model_version, question_def, input_text)`.

- **Jev**
  - Uses `AsyncTypeSafeClient` with a concurrency semaphore of 8 and one `Choice` question per example. `instructions` is the task description; `criteria` maps each class label to a one-sentence definition.
  - The model version is pinned in config (it is read from the first `make smoke` run), not `jev-latest`, so results stay reproducible.
  - All SDK specifics live in `backends/jev.py`, so API drift touches one file.
  - The CLI loads `.env` at start-up with `python-dotenv`; an already-exported `TYPESAFE_API_KEY` takes precedence. The key is passed only to the SDK client. It is never logged or cached, and error messages show at most its last 4 characters.
  - SDK retries are disabled (`max_retries=0`), so every HTTP request is exactly one counted inference. A failed request still counts against the cap but is not cached, so a later run may retry it while the cap has headroom. An item still without a response when the experiments run is treated as `valid=False` with a uniform distribution and counts as an error.
- **Mock ("mock-jev")**
  - Deterministic and seeded, with no trained model. For each item it builds logits from the gold label (a configurable margin on the gold class), adds seeded Gaussian noise, and applies a temperature, giving a plausible, imperfectly calibrated model. Noise is seeded per item ID, so outputs do not depend on call order.
  - It also implements the masking-based probability function used by explainability: removing a feature perturbs the logits by a seeded, feature-specific amount, so attributions are non-trivial and reproducible.
  - Every output has `meta.mode = "mock"`.

### D5. Jev inference budget (≤ 100 in total)
- **Allocation** (`configs/budgets.yaml`): `smoke` 1, `predict` 99 (39 val + 60 test), `explain` 0, every other stage 0. The global cap `jev_total: 100` sits above the per-stage budgets.
- **Global, persistent ledger.** Every live Jev request is recorded in a ledger table in the sqlite cache before it is sent, whether or not it succeeds. Before a stage makes calls, the dry-run count plus the ledger total is checked against `jev_total`; if it would exceed 100, the stage stops before any call and reports required vs allowed. The cap therefore holds across separate CLI invocations, reruns and crashes.
- **Reruns** are served from the cache and add nothing to the ledger. After a failure, a rerun retries only the missing items, and only while the ledger has headroom.
- `make predict MODE=live` always prints the dry-run report (calls needed, ledger total, cap) first.

### D6. Metrics
- Built on scikit-learn metrics.
- ECE uses 15 equal-width bins on max-probability confidence.
- Bootstrap CIs use 1,000 resamples with seed 0.

### D7. Thresholding and genetic algorithm
- **Decision rule:** predict `argmax_k w_k · p_k` with `w ∈ R_+^K` and `w_1 = 1` fixed. This is a threshold for binary tasks and per-class weights for multiclass.
- **Bayes-optimal reference:** `argmin_j Σ_i p_i C[i,j]`.
- **GA:** a small NumPy implementation with a population of 60, 80 generations, tournament selection, blend crossover, Gaussian mutation in log-space, elitism of 2 and seed 0. It minimizes mean validation cost.
  - We use our own GA rather than DEAP/pygad because it is about 60 lines, transparent for the paper, and has no dependency risk.
- **Cost matrix:** German Credit's canonical 5:1 matrix, in `costs.yaml`.

### D8. Triage cost model
- Confidence is the maximum class probability. Jev's confidence statistic `(K·p_max − 1)/(K − 1)` is an affine transform of `p_max` for a fixed K, so it gives the same ranking. We state this in the paper rather than duplicating curves.
- The only escalation target is a simulated human-in-the-loop reviewer.
- All costs are in USD: `total = Jev inference $ + human escalation $ + λ · misclassification-cost-units`.
  - λ (dollars per cost unit) is configurable, with a default of $1.
  - The human cost defaults to $0.50 per item.
  - A sensitivity plot varies λ and the human cost over one order of magnitude, so conclusions do not rest on arbitrary constants.
- Risk–coverage curves average over tied confidences (the expected curve under random tie-breaking), since Jev reports its confidence rounded.
- K is swept over the quantiles of Jev confidence on the Jev subset. The optimal K is chosen on the 39 val items and reported on the 60 test items.

### D9. Importance sampling
- **Pool:** the 60-item Jev test subset; its full-pool metric is the ground truth.
- **Proposal:** `q_i ∝ (1 − ε)(1 − p_max,i)^α + ε/N`, with ε = 0.1 and α = 1 by default, plus an α-sensitivity analysis. This approximates the loss-proportional proposal of Active Testing / Active Risk Estimation (Kossen et al. 2021; Sawade et al. 2010).
- **Sampling:** with replacement, using the Hansen–Hurwitz estimator for the error rate (the quantity the loss-proportional proposal targets); accuracy is one minus it. F1 uses the ratio of weighted TP/FP/FN.
- **CIs:** CLT intervals from weighted variance.
- **Trials and budgets:** 1,000 trials per budget, with budgets n ∈ {5, 10, 20, 30}.
- **Unit test:** a synthetic pool checks that the estimator is unbiased to within Monte Carlo error.

### D10. Explainability
- **Mock-only.** The pipeline runs on mock-jev and makes zero live Jev calls (`explain` budget 0). It shows that a class-probability function is all KernelSHAP/LIME need; it does not produce live Jev attributions.
- **Examples:** 20 seeded German Credit examples from the Jev test subset.
- **Masking:** features are removed from the serialization entirely. This is cleaner than an "unknown" token, and the prompt says missing fields are unknown.
- **Methods:**
  - KernelSHAP (`shap.KernelExplainer`) with 200 coalitions per example (`nsamples`). Its background is the empty-coalition "all features removed" input, which gives an exact base value.
  - LIME (`lime`) with 200 samples.
  - Both call the cached, batched probability function.
- **Faithfulness:** delete the top-k features, with k ∈ {1, 3, 5}, re-query the model, and compare the probability drop and flip rate against random-k deletion (5 random draws).
- **Agreement:** Kendall τ and top-3 overlap between KernelSHAP and LIME.
- **Call cost:** the stage reports how many model calls one explanation needs (about 200 per method per example). The paper uses this to explain why live attributions were out of scope under the 100-inference cap.

### D11. Paper build
- The paper is `paper/main.tex`, using `article` with arXiv-friendly settings; `sections/*.tex` are hand-written prose.
- `make paper` runs `python -m ciayn paper --mode live`. This writes `paper/generated/*.tex` (booktabs tables and `\newcommand` macros for inline numbers) and PDF figures, then runs `latexmk -pdf`.
- If any consumed result has `mode = mock`, or `--mode mock` is passed, a generated `\def\mockdraft{}` switches on the `draftwatermark` package with "DRAFT — MOCK RESULTS".
- The live paper never reads `results/mock/**`. Its explainability section uses only the per-explanation call counts from the explain config, which are not model outputs.
- Inline numbers in prose use generated macros (for example `\JevGermanMacroFone`), so no number is typed by hand.

### D12. Testing
Pytest covers:
- serialization (the label never appears)
- split determinism
- cache warm-rerun (zero calls; identical output, via a counting fake backend)
- probability-vector invariants
- GA seeded reproducibility and its match with the analytic threshold on synthetic calibrated data
- IS unbiasedness
- the budget guard and the persistent 100-inference Jev cap (a counting fake SDK shows the cap holds across separate invocations and that failed requests are counted)
- the mock-watermark switch

Nothing in the test suite makes network calls.

## Risks / Trade-offs

- **Public Jev docs are third-party and may be inaccurate.** → All SDK use sits in `backends/jev.py`. The smoke test (`make smoke`) validates the response shape before a live run and fails loudly on drift.
- **No baseline model.** → The paper claims what tooling a class distribution unlocks, not that Jev beats anything; this is stated explicitly.
- **Tiny samples (39 val, 60 test, one dataset) under the 100-inference cap.** → Bootstrap CIs are shown everywhere, claims are phrased as demonstrations of tooling rather than precise estimates, and the Limitations section says so.
- **Memorization or contamination.** German Credit is famous, and LLMs may have memorized it. → This is noted in the Limitations section; it cannot be controlled.
- **A failed or interrupted live run burns part of the 100 inferences.** → Requests are ledgered before they are sent, retries are disabled, and reruns only fill missing items within the remaining headroom. `make smoke` validates the response shape with a single call before `predict`.
- **Label or option-order effects in Jev Choice.** → Class order in `criteria` is fixed and documented. Shuffling it is out of scope and noted as a limitation.
- **API key leakage.** → The key lives only in the gitignored `.env` (mode 600). It is excluded from cache keys, `meta` and logs, and a pytest checks that it never appears in cache or result files.
- **Mock results might be mistaken for real ones.** → Separate results directories, a `mode` field in every artifact, and a watermark in the paper.
- **Explainability results are mock-only.** → They are never presented as Jev findings; the live paper shows the method and its call cost only.

## Migration Plan

Not applicable (greenfield). Rollout order:
1. Setup and mock mode end-to-end.
2. The Jev key is in `.env` (done), and `make smoke` passes.
3. Live predictions (at most 99 Jev calls, after the dry-run report).
4. Live experiments.
5. Paper build.

## Open Questions

- The exact Jev model version to pin, which will be read from the first smoke-test response.
- The final dollar scale λ and the human-review cost used in headline triage numbers. The sensitivity analysis makes this safe to settle later.
