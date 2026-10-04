# Tasks

## 1. Record what was spent on Banking77

- [x] 1.1 Copy `paper/`, `results/live/banking77/` and `results/splits/` outside the working copy and save the current `paper/generated/macros.tex` as the comparison snapshot; verify the copies exist and list their sizes.
- [x] 1.2 Add `provenance --removed banking77`, which writes `results/live/removed_datasets.json` (Jev requests by stage, tokens and cost; dedicated AWS executions and cost; shared Qwen smoke runs with their whole cost); verify with a test on a fake ledger, request log and attempts record, and that the live file's Jev requests by stage add up to the request-log rows for Banking77.
- [x] 1.3 Make the cleanup refuse to run without that file; verify with a test.

## 2. Remove Banking77 from the study

- [x] 2.1 Remove Banking77 from `config.DATASETS` and every configuration file, delete its loader and pinned-source configuration, its dataset document and its prior-work rows; verify `grep -ri banking src configs docs/datasets` finds only the removed-share accounting.
- [x] 2.2 Remove the Banking77 line from the manifest without touching the header or the other lines, delete `results/splits/banking77.json` and write `results/splits/REMOVED.json`; verify with a test and on the live files that the four remaining hashes equal the recorded ones and every stage still passes `manifest.verify()`.
- [x] 2.3 Remove the Banking77-only code paths (Qwen `engine: none` and `not_scored_reason`, the `not run` table cells, the Banking77 escalation-validation entry) while keeping the generic "not available" path; verify with the existing missing-model tests.
- [x] 2.4 Delete `results/live/banking77/` and `results/mock/banking77/`; verify no stage output or generated file references them.
- [x] 2.5 Rewrite the tests that used Banking77 on AG News, keeping one synthetic many-option test for prefix-option scoring; verify the full suite passes offline.
- [x] 2.6 Rerun `metrics`, `thresholding`, `importance`, `comparison` and `escalation` in live mode, one at a time; verify against the snapshot that every macro of the four remaining datasets is unchanged and list the macros that differ (dataset counts, five-dataset totals, removed-share macros).
- [x] 2.7 Add the removed-share macros and the extra row of the cost appendix to the generator; verify the Jev ledger total and the recorded AWS cost equal their values before the removal.

## 3. NeurIPS 2026 format

- [x] 3.1 Add the official `neurips_2026.sty` and `paper/STYLE_SOURCE.json` (URL, date, sha256) and switch `main.tex` to it (no `geometry`, natbib options passed to the style); verify the paper builds and the PDF shows line numbers and an anonymous author block.
- [x] 3.2 Generate the merged main-text metrics table and the compact main-text tables for the narrower text width, keeping the full tables for the appendices; verify each fits the text width without scaling below `\small`.
- [x] 3.3 Restructure the sections to the page budget of design D6: condense the text, move the listed tables, figures and paragraphs to appendices and cite each from the main text; verify with a script that every table and figure of the previous build for the four datasets is included somewhere and referenced.
- [x] 3.4 Remove Banking77 from every section, caption and generated sentence, and state once in the setup and in the cost appendix that a fifth dataset was run and removed, with its share of the costs; verify `pdftotext` finds Banking77 only in those two places.
- [x] 3.5 Add the NeurIPS checklist with every question answered and justified from the paper and the saved outputs; verify no placeholder text of the kit remains.
- [x] 3.6 Remove identifying text (account number, bucket and role names) from the paper and add `configs/submission.yaml` with the strings to search for; verify none appears in the PDF text or metadata.

## 4. Format checks and final verification

- [x] 4.1 Add the format checks to `verify-paper` (main text ends by page nine, fonts embedded and not Type 3, checklist complete, style hash, identifying strings absent) and write their results to `paper_check.json`; verify with tests that each violation fails the stage and names the requirement.
- [x] 4.2 Build the live paper and run `verify-paper`; verify every number matches the saved outputs and every format check passes, tightening text in the order given in design D6 if the main text exceeds nine pages.
- [x] 4.3 Inspect every page of the rendered PDF (line numbers, equations, table and figure legibility, appendix references) and update `docs/PDF_INSPECTION.md`; verify the LaTeX log has no overfull box above the tolerance and no unresolved reference.
- [x] 4.4 Regenerate `docs/REPORT.md` for four datasets with the removed-dataset note, and update `README.md`, `docs/AWS_BASELINE.md` and `docs/JEV_ACCESS.md`; verify `scripts/fresh_checkout_check.sh` passes with zero Jev requests and zero SageMaker jobs.
- [x] 4.5 Run the mock pipeline and build the mock paper in the new format; verify the watermark is present and the page-limit and checklist checks pass.
