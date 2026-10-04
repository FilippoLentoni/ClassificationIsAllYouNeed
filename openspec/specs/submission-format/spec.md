# submission-format Specification

## Purpose

Defines the NeurIPS 2026 submission requirements the compiled paper must meet, and the automated check that enforces them, so that the paper can be submitted to the main track without format changes.

## Requirements

### Requirement: NeurIPS 2026 style
The paper SHALL be typeset with the official, unmodified `neurips_2026` style file in its default main-track submission mode, on US Letter paper, and SHALL NOT change the style's margins, font sizes or spacing. The style file SHALL be stored with the paper together with a record of where and when it was downloaded and its sha256.

#### Scenario: Official style unmodified
- **WHEN** the paper is built
- **THEN** it loads `neurips_2026.sty` whose sha256 equals the recorded hash of the downloaded file, and no package or command overrides the page geometry or the base font size

### Requirement: Anonymous submission
The compiled PDF SHALL be anonymous: no author names, affiliations, e-mail addresses or acknowledgements, and no text, URL, account identifier, bucket name or file metadata that identifies the authors. Line numbers SHALL be present.

#### Scenario: No identifying text
- **WHEN** the format check runs on the compiled PDF and its metadata
- **THEN** it finds none of the configured identifying strings (author name, e-mail address, AWS account number, bucket name, home directory) and reports the strings it searched for

### Requirement: Page limit
The main text, from the title to the end of the conclusion and including every figure and table placed in it, SHALL be at most nine pages. References, appendices and the checklist SHALL start after the main text and do not count.

#### Scenario: Main text within nine pages
- **WHEN** the format check runs
- **THEN** it reports the page on which the references start and fails if the main text ends after page nine

### Requirement: Paper checklist
The paper SHALL end with the NeurIPS paper checklist from the official kit, with every question answered Yes, No or NA and a justification, and each answer SHALL be consistent with what the paper and the saved outputs contain.

#### Scenario: Checklist complete
- **WHEN** the format check runs
- **THEN** every checklist question has an answer and a justification, and none still contains the kit's placeholder text

### Requirement: Fonts and figures
Every font in the PDF SHALL be embedded and none SHALL be Type 3. Figures and tables SHALL fit within the text width and remain legible at the style's font sizes.

#### Scenario: Fonts embedded
- **WHEN** the format check lists the fonts of the compiled PDF
- **THEN** every font is embedded and none is Type 3

#### Scenario: No overflow
- **WHEN** the paper is built
- **THEN** the LaTeX log has no overfull horizontal box wider than the configured tolerance, and the page-by-page inspection record confirms that every table and figure is legible

### Requirement: Format check in the verification step
The verification step that checks the paper's numbers SHALL also run the format checks above and SHALL fail, naming the violated requirement, if any of them fails. The checks SHALL NOT apply to a mock draft's watermark, but SHALL apply to its page limit and checklist.

#### Scenario: Violation stops verification
- **WHEN** the main text exceeds nine pages or a checklist answer is missing
- **THEN** the verification step fails and names the requirement
