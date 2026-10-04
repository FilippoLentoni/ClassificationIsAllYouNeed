# PDF inspection

`paper/main.pdf`, live build of 2026-10-01 in the NeurIPS 2026 anonymous submission format
(33 pages, US Letter), rendered page by page and read after Banking77 was removed from the study.
`verify-paper --mode live` passed on the same build: every generated file matches a fresh
regeneration from the saved outputs, every value used in the text was found in the PDF, and the
seven format checks passed (style file hash, anonymity, page limit, checklist, embedded fonts, no
overflow, every table and figure included and cited). The LaTeX log has no overfull box and no
unresolved reference.

| Pages | Content | Checked |
|---|---|---|
| 1-2 | Title, anonymous author block, abstract, introduction, background, related work | Line numbers present; the two displayed definitions read correctly with them. |
| 2-4 | Setup, datasets table, metrics table, macro-F1 curves | Four datasets, four models per dataset; no "not run" cell. |
| 5-6 | Cost table, escalation figure and table, threshold table | Tables fit the 5.5-inch text width without scaling below `\small`. |
| 7-8 | Threshold figure, importance-sampling table, explanations table, limitations, conclusion | Main text ends on page 8 of 9 allowed. |
| 8-10 | References | All citations resolve. |
| 11-12 | Appendices A-B: prior evaluations; protocol, label budgets, costs, removed dataset, societal impact | Banking77 appears only here: one paragraph and one row of the cost table. |
| 13-16 | Appendices C-E: full metrics, calibration, reliability, learning curves, thinking, escalation items | Vertical stretching seen in the first render was removed. |
| 16-19 | Appendices F-H: thresholds, importance sampling, explanations in full | A sentence that lost its subject when it moved to Appendix D was fixed. |
| 20-25 | Appendices I-J: logged Jev requests and responses; Qwen prompts | Four examples each. |
| 25-26 | Appendices K-L: dataset documentation, per-class metrics | Fit the page. |
| - | NeurIPS paper checklist | Removed on 2026-10-02 at the author's request; the paper now has 26 pages. |

The mock build (29 pages) was also rendered: the "DRAFT - MOCK RESULTS" watermark is on every
page, and its page-limit and checklist checks pass.
