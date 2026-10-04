"""`verify-paper`: check every reported number against the saved experiment outputs (design D11).

Two checks:

1. Every generated file (macros, tables, appendices, figures) is regenerated from the saved
   outputs into a temporary directory and compared with `paper/generated/`. A difference means
   the paper was built from something other than the saved outputs.
2. Every number macro used in the paper's text is looked up in the text extracted from the
   compiled PDF, so a value that is defined but does not reach the page is reported.

The result, with the source file of every macro, is written to results/<mode>/paper_check.json.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import build, submission


def plain(value: str) -> str:
    """The text a LaTeX macro value produces on the page."""
    v = value.replace("{,}", ",").replace(r"\textbackslash{}", "\\")
    v = re.sub(r"\\([_%&#$\{\}])", r"\1", v)
    if v.startswith("$") and v.endswith("$"):          # simple inline math such as $10^{-12}$ prints as 10-12
        v = re.sub(r"[$^{}]", "", v)
    return v


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\u2212", "-").replace("\u2010", "-").replace("\u2013", "-"))


def in_text(value: str, pdf_text: str) -> bool:
    """True if the value appears on the page, allowing for line breaks and end-of-line hyphenation."""
    target = squash(plain(value))
    if not target:
        return True
    variants = (squash(pdf_text), squash(pdf_text.replace("-\n", "")))
    if any(target in v for v in variants):
        return True
    return target.replace("-", "") in squash(pdf_text).replace("-", "")


def macros_used(paper: Path) -> set[str]:
    files = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]
    text = "".join(p.read_text() for p in files if p.exists())
    return set(re.findall(r"\\([A-Za-z]+)", text))


def compare_generated(mode: str, generated: Path) -> list[str]:
    """Names of generated files that differ from a fresh regeneration out of the saved outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "generated"
        build.generate(mode, out=fresh)
        names = sorted({str(p.relative_to(fresh)) for p in fresh.rglob("*") if p.is_file()} |
                       {str(p.relative_to(generated)) for p in generated.rglob("*") if p.is_file()})
        return [n for n in names if not (fresh / n).exists() or not (generated / n).exists()
                or (fresh / n).read_bytes() != (generated / n).read_bytes()]


def verify(mode: str, check_pdf: bool = True) -> dict:
    generated = config.PAPER / "generated"
    built = json.loads((generated / "mode.json").read_text())
    if built["mode"] != mode:
        raise RuntimeError(f"paper/generated was built in {built['mode']} mode, not {mode}; run `python -m ciayn paper --mode {mode}`")
    different = compare_generated(mode, generated)
    sources = json.loads((generated / "macro_sources.json").read_text())
    used = macros_used(config.PAPER) & set(sources)
    pdf_text = None
    if check_pdf:
        pdf = config.PAPER / "main.pdf"
        if not pdf.exists():
            raise RuntimeError(f"{pdf} does not exist; build the paper first")
        pdf_text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    rows = [{"macro": name, "value": plain(info["value"]), "source": info["source"], "used_in_text": name in used,
             "found_in_pdf": (in_text(info["value"], pdf_text) if pdf_text is not None and name in used else None)}
            for name, info in sorted(sources.items())]
    missing = [r["macro"] for r in rows if r["found_in_pdf"] is False]
    report = {"checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": mode,
              "mock_draft": built["mockdraft"], "generated_files_match_saved_outputs": not different,
              "generated_files_that_differ": different, "macros_defined": len(rows), "macros_used_in_text": len(used),
              "pdf_checked": pdf_text is not None, "macros_used_but_not_found_in_pdf": missing,
              "format": submission.run_checks(mode, pdf_text) if pdf_text is not None else {}, "macros": rows}
    (config.results_dir(mode) / "paper_check.json").write_text(json.dumps(report, indent=1))
    return report


def run(args) -> int:
    report = verify(args.mode, check_pdf=not args.no_build)
    print(f"verify-paper ({args.mode}): {report['macros_defined']} generated numbers with their source files; "
          f"{report['macros_used_in_text']} used in the text")
    problems = []
    if report["generated_files_that_differ"]:
        problems.append("generated files differ from the saved outputs: " + ", ".join(report["generated_files_that_differ"]))
    if report["macros_used_but_not_found_in_pdf"]:
        problems.append("values used in the text but not found in the PDF: " + ", ".join(report["macros_used_but_not_found_in_pdf"]))
    for name, check in report["format"].items():
        if not check["ok"]:
            detail = {k: v for k, v in check.items() if k not in ("ok", "requirement", "searched_for")}
            problems.append(f"submission format, requirement \"{check['requirement']}\" ({name}): {detail}")
    if problems:
        raise RuntimeError("paper check failed: " + "; ".join(problems))
    print(f"verify-paper ({args.mode}): every generated file matches a fresh regeneration from the saved outputs"
          + ("; every value used in the text was found in the PDF" if report["pdf_checked"] else "; PDF not checked (--no-build)"))
    if report["format"]:
        f = report["format"]
        print(f"verify-paper ({args.mode}): submission format: {len(f)} checks passed; main text ends on page "
              f"{f['page_limit']['main_text_ends_on_page']} of {f['page_limit']['limit']}; checklist " +
              (f"{f['checklist']['answered']} of {f['checklist']['questions']} answered" if "answered" in f["checklist"]
               else "left out on purpose (not a complete NeurIPS submission)"))
    return 0
