# Design

## Context

See proposal.md for the motivation. Facts that shape the approach, observed on 2026-10-01:

- `config.DATASETS` drives every stage and the paper generator; 35 source, configuration, test and
  section files mention Banking77. Code that exists only for it: the GitHub CSV loader in
  `data/datasets.py`, the `not_scored_reason` / `engine: none` path in `llm/`, and the
  `not run` table cells in `paper/build.py`.
- `results/splits/MANIFEST.sha256` has one hash line per dataset and a `# frozen <time>` header;
  `manifest.verify()` checks one dataset at a time, so the four remaining entries are independent
  of the fifth.
- The request log records the dataset of every request, so the Jev requests, tokens and cost that
  belong to Banking77 can be computed exactly; the 100 requests from before logging were German
  Credit only. Banking77's XGBoost (2) and BERT (1) executions are separate records in
  `baseline_attempts.json`; the Qwen smoke runs that included Banking77 items are shared jobs.
- The response cache, ledger and request log are not tracked and are needed to compute that share;
  they are present on this machine and backed up in the project bucket.
- The official kit (`neurips_2026.sty`, `neurips_2026.tex`, `checklist.tex`) downloads from
  `media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`. The style
  sets a 5.5 x 9 inch text block at 10pt, loads natbib, adds line numbers and anonymises the
  author block by default; the kit states a nine-page limit for content, with references,
  checklist and appendices not counted.
- The current paper is a 31-page `article` at 1-inch margins: about 15 pages of main text, with a
  6.5-inch text width. Several tables rely on `\fittable` to scale to the text width.
- `verify-paper` already regenerates every generated file, extracts the PDF text and checks that
  every used macro value appears in it.

## Goals / Non-Goals

**Goals:**
- A four-dataset study in which nothing computed for the remaining datasets changes.
- A PDF that meets the NeurIPS 2026 main-track submission format, checked automatically.
- Every result of the four datasets still in the paper, in the main text or an appendix.

**Non-Goals:**
- No new experiment, Jev request or SageMaker job; no change to models, templates or thresholds.
- No camera-ready or preprint variant, and no rewriting of the argument: text is condensed, not
  re-argued.
- No deletion from the response cache, ledger or request log.

## Decisions

### D1. Save what was spent on Banking77 before deleting anything
A one-off stage, `provenance --removed banking77`, writes `results/live/removed_datasets.json`
from the ledger, request log and attempts record: Jev requests by stage, input tokens and cost;
AWS executions dedicated to Banking77 with their cost; and the shared Qwen smoke runs that
included Banking77 items, listed with their whole cost and labelled as shared. The file is
tracked, so the paper and report rebuild without the cache. Deleting Banking77 results is a later
task and refuses to run if this file is missing.
*Alternative:* recompute the share at build time; rejected because the paper must rebuild from
tracked files alone.

### D2. Remove the dataset from configuration; keep the frozen files of the others untouched
`config.DATASETS` becomes the four names. `results/splits/banking77.json` is deleted and its line
removed from the manifest by a function that leaves the `# frozen` header and the other four lines
byte-identical and writes `results/splits/REMOVED.json` (dataset, date, the four unchanged
hashes, the removed hash). `freeze` is not run. A test asserts the four hashes equal the recorded
ones.
*Alternative:* `freeze --refreeze` for four datasets; rejected because it would rewrite the
freeze time and break the claim that identifiers were frozen before any use.

### D3. Delete Banking77-only code and results
Removed: the Banking77 loader and its pinned-CSV configuration; `engine: none` and
`not_scored_reason` in `llm/` and the `not run` cells in the paper generator (the generic
"not available" path stays, because a missing model result must still be reportable);
Banking77's `escalation` configuration entry; `results/live/banking77/`, `results/mock/banking77/`;
`docs/datasets/banking77.md`; the Banking77 rows of `prior_work.yaml`. The 500 Banking77
escalation-validation predictions are deleted with the results and counted in D1. Tests that used
Banking77 as the many-class case move to AG News; one synthetic many-option test keeps the
prefix-option scoring covered.

### D4. Totals keep everything; the removed share is shown next to them
`provenance` totals are unchanged (ledger and attempts are not edited). The generator adds macros
for the removed share (requests, tokens, Jev cost, dedicated AWS cost, shared Qwen smoke cost) and
the cost appendix gains one row. The per-stage and per-model figures that the main text quotes
stay whole-study figures, with the removed share stated once in the setup and in the appendix.

### D5. NeurIPS style
`paper/neurips_2026.sty` is the downloaded file, unmodified, with `paper/STYLE_SOURCE.json`
(URL, date, sha256). `main.tex` uses `\documentclass{article}` and `\usepackage{neurips_2026}`
(default: main track, anonymous, line numbers); `geometry` is removed; natbib options go through
`\PassOptionsToPackage{numbers,sort&compress}{natbib}`. `\author` stays a placeholder that the
style hides. The mock watermark keeps working through `draftwatermark`.

### D6. Nine pages: what stays and what moves
Main text budget, in pages of the NeurIPS layout:

| Part | Pages | Kept in the main text | Moved to an appendix |
|---|---|---|---|
| Abstract, introduction | 1.2 | contributions as prose | |
| Background | 0.7 | definition and the list of tools | the worked sampling-variance remarks |
| Related evaluations | 0.3 | two sentences | prior-evaluation table |
| Setup | 1.3 | datasets table, one paragraph per model | label-budget table, cost and budget paragraph, uncertainty details |
| Metrics and calibration | 1.5 | one merged table (accuracy, macro-F1, ROC-AUC, ECE) | PR-AUC, Brier, log-loss, confidence table; reliability diagrams |
| Labels, cost, latency | 1.2 | macro-F1 curves, cost table | accuracy curves, thinking paragraph and table |
| Escalation | 0.7 | table | curve figure |
| Thresholds | 0.7 | Jev and XGBoost table | sweep figure, all-model table |
| Importance sampling | 0.5 | finding and the variance ratios | RMSE table, figure, the two explanations in full |
| Explanations | 0.5 | finding and the deletion-test numbers | table, both figures, cost |
| Limitations, conclusion | 0.4 | | |

The generator produces the merged main-text metrics table and keeps the existing full tables for
the appendix; nothing generated is removed except Banking77 rows. Each moved item is cited from
the main text ("Appendix X, Table Y"). If the build exceeds nine pages, text is tightened in the
order: background, setup, importance sampling, explanations; results tables are not shrunk below
`\small`.
*Alternative:* drop importance sampling and explanations from the main text entirely; rejected by
the user's choice to keep all results visible.

### D7. Checklist
`paper/sections/checklist.tex` is the kit's checklist with each of its questions answered and
justified from what the paper contains (for example: code and data availability, compute
resources, limitations, licences of the datasets and models). Answers that depend on a decision
not made here (release of code, a public URL) are answered conservatively ("No" or "NA" with the
reason) rather than promised.

### D8. Format checks inside `verify-paper`
`paper/verify.py` gains `format_checks()`: the page on which the main text ends, read from a
label placed after the conclusion (fail if it is after page nine); `pdffonts` (every font
embedded, none Type 3); checklist present, every question answered, no kit placeholder; the
recorded sha256 of the style file; the configured identifying strings absent from the PDF text
and metadata (`configs/submission.yaml`: author name, e-mail addresses, AWS account number,
bucket name, home directory, the user name of this machine). Results go into
`results/<mode>/paper_check.json`; any failure fails the stage and names the requirement. The
mock build is checked for page limit and checklist only.

### D9. Anonymity in the text
The setup and appendices may name instance types, regions and container versions, which do not
identify the authors, but not the account number, bucket or role names. The request examples in
Appendix A show the public API URL and no headers; they are kept. Self-identifying phrasing
("our bucket") is fine; identifiers are not.

## Risks / Trade-offs

- [Nine pages may not hold all eleven parts] → the budget above; the order in which text is
  tightened is fixed; the check fails the build rather than letting the limit slip.
- [Narrower text width makes wide tables illegible when scaled] → the main text gets purpose-built
  compact tables; full tables go to appendices where `\small` or landscape is acceptable.
- [Removing a dataset could silently change a remaining result] → a snapshot of every macro of
  the four datasets is taken before the change and compared after it; only dataset counts, totals
  that summed five datasets and the removed-share macros may differ.
- [The share spent on Banking77 cannot be recomputed after cleanup] → D1 runs first and its
  output is tracked.
- [Line numbers from the style can clash with display equations] → the two equations are wrapped
  as the style's documentation recommends, and the inspection record covers them.
- [Checklist answers overclaim] → each answer cites the section or file that supports it; D7.

## Migration Plan

1. Save the removed share (D1) and a snapshot of the current macros.
2. Remove Banking77 from configuration, code, results, splits and tests (D2, D3); rerun the
   offline stages; compare the snapshot.
3. Switch the style, restructure the sections and appendices, add the checklist (D5-D7).
4. Add the format checks, build, verify, inspect every page, regenerate the report (D8).
Rollback: the previous paper and results are in the archived change's state of the working copy;
there are no git commits in this project, so a copy of `paper/` and `results/live/banking77/` is
kept outside the working copy before step 2.

## Open Questions

- Whether the submission will include code; the checklist answers "No" until the user decides.
