# Proposal

## Why

Banking77 is the one dataset on which the four-model comparison is incomplete (Qwen3.6-27B was not
run on it), and the paper is typeset as a plain 31-page article. The user wants a four-dataset
study with every model on every dataset, in the NeurIPS 2026 submission format.

## What Changes

- **BREAKING: Banking77 is removed from the study.** The study covers German Credit, Breast Cancer
  Wisconsin Diagnostic, SST-2 and AG News. Banking77 leaves the dataset configuration, every
  stage, table, figure, macro, appendix and sentence, the dataset documentation and the prior-work
  table; its saved results, split file and the code that exists only for it are deleted.
- **What was spent on Banking77 stays counted and is stated.** The Jev requests already made on it
  (predictions, explanations and the 500 escalation validation items) remain in the ledger and in
  the Jev total; its XGBoost and BERT executions and the Qwen smoke runs that measured it remain
  in the AWS total. The paper and the report say how many requests and how much cost belong to the
  removed dataset. No Jev request and no SageMaker job is needed for this change.
- **The frozen identifiers of the four remaining datasets do not change.** Their split files stay
  byte-identical; only the manifest's entry for Banking77 is removed, and that removal is recorded.
- **NeurIPS 2026 anonymous submission format.** The paper uses the official `neurips_2026` style
  (main track, double-blind, line numbers), with a main text of at most nine pages including
  figures and tables, followed by references, the appendices and the NeurIPS paper checklist.
  Anything that identifies the authors or their AWS account is removed from the paper.
- **Nothing is cut from the paper to fit; material moves.** The main text keeps the framing, the
  setup, metrics and calibration, the four-model comparison, escalation and thresholds, and a
  condensed account of importance sampling and explanations. Their detailed tables and figures,
  the protocol details and the cost accounting move to the appendices, which the main text cites.
  The main text must stand on its own.
- The page-limit and format checks become part of `verify-paper`: it fails if the main text
  exceeds nine pages, if the checklist is missing, or if a font is not embedded.

## Capabilities

### New Capabilities
- `submission-format`: the NeurIPS 2026 submission requirements the compiled paper must meet (style, anonymity, page limit, checklist, fonts, paper size) and their automated check.

### Modified Capabilities
- `experiment-harness`: four datasets instead of five; frozen identifiers survive the removal of a dataset; no further Jev requests; spent-but-removed work stays accounted.
- `classification-evaluation`: the multiclass and held-out scenarios name the remaining datasets.
- `ensemble-escalation`: validation items come from AG News only among the multiclass datasets.
- `fine-tuned-encoder-baseline`: four datasets.
- `supervised-baseline`: four datasets on one combination.
- `perturbation-explainability`: four datasets.
- `paper-manuscript`: structure and length follow the NeurIPS format; four datasets; four request and response examples; provenance states what was spent on the removed dataset.

## Impact

- **Configuration:** `configs/datasets.yaml`, `encoder.yaml`, `llm.yaml`, `escalation.yaml`, `prior_work.yaml`, `budgets.yaml` (comments), `report_notes.yaml`.
- **Code:** `config.DATASETS`; `data/datasets.py` (Banking77 loader), `escalation_val.py`, `llm/prompts.py` and `llm/cli.py` (the not-run path), `paper/build.py`, `paper/report.py`, `paper/verify.py`, `paper/provenance.py` (cost of the removed dataset).
- **Results:** `results/live/banking77/`, `results/mock/banking77/`, `results/splits/banking77.json` deleted; `results/splits/MANIFEST.sha256` loses one entry; a record of the removal is added.
- **Paper:** `paper/main.tex` on `neurips_2026.sty`; every section; new `checklist.tex`; appendices reorganised.
- **Docs and tests:** `docs/datasets/banking77.md` removed; README, REPORT, AWS and Jev documents updated; tests that use Banking77 rewritten on AG News.
- **No remote calls:** zero Jev requests, zero SageMaker jobs. The response cache and request log keep their Banking77 entries.
