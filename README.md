# Classification Is All You Need

Experiments and paper on what a full class-probability distribution makes possible, using
TypeSafe's Jev (a model that returns a probability for every option of a fixed label set)
zero-shot on four public datasets, next to a second zero-shot model (self-hosted Qwen3.6-27B)
and two supervised models (XGBoost and fine-tuned BERT), all run on Amazon SageMaker.

- **Datasets:** German Credit, Breast Cancer Wisconsin Diagnostic, SST-2, AG News. A fifth, Banking77, was run
  and removed; what was spent on it is in `results/live/removed_datasets.json` and stays in the totals.
- **Experiments:** metrics and calibration for four models, F1 / cost / latency by training
  labels, escalation of low-confidence Jev predictions to Qwen, threshold selection, accuracy
  estimation by importance sampling, and KernelSHAP / LIME explanations of Jev.
- **Paper:** `paper/main.pdf`, in the NeurIPS 2026 anonymous submission format (official style file in
  `paper/`, main text within nine pages; the NeurIPS checklist was removed at the author's request and is
  kept in `paper/sections/checklist.tex`); every number in it is generated from the
  saved outputs, and `verify-paper` checks both the numbers and the format.
- **Findings, limitations, costs:** [docs/REPORT.md](docs/REPORT.md).

## Layout

```
configs/      datasets, budgets, costs, baseline, experiment settings, prior-work numbers
src/ciayn/    data/ (loaders, splits, manifest), backends/ (Jev, mock, cache, ledger, request log),
              baseline/ (SageMaker pipeline and job scripts), experiments/, paper/
infra/        CloudFormation template for the baseline
results/      splits/ (frozen identifiers), live/ (saved predictions, tables, figures), mock/ (ignored)
cache/        response cache, ledger and request log (ignored; backed up outside the working copy)
paper/        main.tex, sections/, generated/ (written by the paper stage)
docs/         JEV_ACCESS.md, AWS_BASELINE.md, datasets/, REPORT.md, PDF_INSPECTION.md
tests/        no test calls Jev or AWS
```

## Setup

```bash
make setup     # installs uv if needed, Python 3.11 and the pinned dependencies
make test      # the whole suite runs offline
```

## Run everything offline (mock mode)

Mock mode replaces Jev and the baseline with clearly labelled synthetic predictors. It needs
no API key and no AWS access, and every artifact it writes is marked as mock.

```bash
uv run python -m ciayn template --mode mock
uv run python -m ciayn predict --mode mock
uv run python -m ciayn metrics --mode mock
uv run python -m ciayn thresholding --mode mock
uv run python -m ciayn importance --mode mock
uv run python -m ciayn explain --mode mock
uv run python -m ciayn escalation-val --mode mock
uv run python -m ciayn comparison --mode mock
uv run python -m ciayn escalation --mode mock
uv run python -m ciayn paper --mode mock         # watermarked draft: DRAFT - MOCK RESULTS
```

Add `--dataset german` (or `wdbc`, `sst2`, `agnews`) to any stage to run one dataset.

## Reproduce the paper from the saved outputs

The live predictions and results are tracked in `results/live/`. These stages read only those
files: no Jev request, no SageMaker job.

```bash
uv run python -m ciayn metrics --mode live
uv run python -m ciayn thresholding --mode live
uv run python -m ciayn importance --mode live
uv run python -m ciayn comparison --mode live      # F1 by training labels, cost and latency
uv run python -m ciayn escalation --mode live      # Jev -> Qwen escalation, from saved predictions
uv run python -m ciayn paper --mode live           # or: make paper
uv run python -m ciayn verify-paper --mode live    # checks every reported number against the saved outputs
uv run python -m ciayn report --mode live          # docs/REPORT.md from the same saved outputs
scripts/fresh_checkout_check.sh                    # the same, from a copy of the tracked files only
```

The explanation results (`results/live/*/explain/`) are tracked too. Re-running the `explain`
stage itself needs the response cache, because each explanation is hundreds of Jev requests.

## Run the study live

Read [docs/JEV_ACCESS.md](docs/JEV_ACCESS.md) (API key, the 60,000-inference cap, request log,
backup) and [docs/AWS_BASELINE.md](docs/AWS_BASELINE.md) (what is created in AWS, cost limit,
teardown) first. Every stage that makes remote calls supports `--dry-run`, which reports what it
would do and stops.

```bash
uv run python -m ciayn freeze                                # once: evaluation identifiers + dataset documentation
uv run python -m ciayn smoke --mode live                     # 1 Jev request
uv run python -m ciayn template --mode live                  # up to 60 requests, then templates are frozen
uv run python -m ciayn predict --mode live                   # Jev predictions for all four datasets
uv run python -m ciayn payload-audit --mode live             # what is known about calls made before request logging
uv run python -m ciayn payload-examples --mode live          # one logged request and response per dataset

uv run python -m ciayn baseline plan --mode live             # what would be created and run in AWS
uv run python -m ciayn baseline deploy --mode live --yes     # only after the template has been approved
uv run python -m ciayn baseline run --mode live              # one SageMaker pipeline execution per dataset
uv run python -m ciayn baseline fetch --mode live            # predictions back, identifiers checked

uv run python -m ciayn encoder run --mode live                # BERT: 5 GPU training jobs
uv run python -m ciayn encoder fetch --mode live
uv run python -m ciayn llm stage --mode live                  # Qwen: copy the pinned weights to S3 once
uv run python -m ciayn llm smoke --mode live --engine transformers   # measure, then project the cost
uv run python -m ciayn llm run --mode live                    # reasoning job, then scoring job, if the projection fits
uv run python -m ciayn llm fetch --mode live
uv run python -m ciayn escalation-val --mode live             # 1,000 Jev requests on validation items

uv run python -m ciayn explain --mode live                   # KernelSHAP, LIME and deletion tests on Jev
uv run python -m ciayn metrics --mode live
uv run python -m ciayn thresholding --mode live
uv run python -m ciayn importance --mode live
uv run python -m ciayn comparison --mode live
uv run python -m ciayn escalation --mode live
uv run python -m ciayn provenance --mode live                # actual usage and cost, saved for the paper
uv run python -m ciayn paper --mode live
uv run python -m ciayn verify-paper --mode live
uv run python -m ciayn report --mode live

uv run python -m ciayn baseline teardown --mode live --yes   # removes the AWS resources
```

## Safeguards

- **Frozen identifiers.** `freeze` writes `results/splits/` with a hash manifest. Every other
  stage checks it and stops if the identifiers or the split configuration changed.
- **AWS spending limit.** Every SageMaker job of every model is checked against one limit before
  it starts; Qwen's full run also needs a cost projection from a measured smoke run to fit.
- **Inference cap.** Every Jev request is counted in a persistent ledger before it is sent.
  Stage budgets and the global cap are checked before any request of a stage is made.
- **Request log.** Every request and response is recorded as it crosses the wire; headers, and
  therefore the API key, are never recorded.
- **No test-label leakage.** Test examples are supplied to the baseline pipeline without labels;
  metrics are computed locally and the two models must be scored on identical identifiers.
- **Mock results cannot pass as live.** Every artifact carries its mode; a paper built from any
  mock artifact is watermarked.
