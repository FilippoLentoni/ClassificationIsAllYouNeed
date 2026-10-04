"""Format checks for the NeurIPS 2026 submission (capability submission-format).

Run by `verify-paper` after the number check. Each check returns {"ok": bool, "detail": ...};
any failure fails the stage and names the requirement. A mock draft is checked for the page limit
and the checklist only.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from .. import config


def _tex(paper: Path) -> str:
    return "".join(p.read_text() for p in [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))] if p.exists())


def style(paper: Path, cfg: dict) -> dict:
    src = json.loads((paper / "STYLE_SOURCE.json").read_text())
    got = hashlib.sha256((paper / cfg["style_file"]).read_bytes()).hexdigest()
    main = re.sub(r"(?m)%.*$", "", (paper / "main.tex").read_text())
    m = re.search(r"\\usepackage(?:\[([^\]]*)\])?\{" + re.escape(cfg["style_file"].removesuffix(".sty")) + r"\}", main)
    options = [o.strip() for o in (m.group(1) or "").split(",") if o.strip()] if m else None
    overrides = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{geometry\}|\\geometry\{|\\setlength\{\\text(?:width|height)\}", main)
    ok = got == src["sha256"] and m is not None and not set(options or []) & set(cfg["forbidden_style_options"]) and not overrides
    return {"ok": ok, "requirement": "NeurIPS 2026 style", "sha256_matches_downloaded_file": got == src["sha256"],
            "style_loaded": m is not None, "options": options, "geometry_overrides": overrides}


def page_limit(paper: Path, cfg: dict) -> dict:
    aux = (paper / "main.aux").read_text() if (paper / "main.aux").exists() else ""
    m = re.search(r"\\newlabel\{end:main\}\{\{[^}]*\}\{(\d+)\}", aux)
    page = int(m.group(1)) if m else None
    return {"ok": page is not None and page <= cfg["page_limit"], "requirement": "Page limit", "main_text_ends_on_page": page,
            "limit": cfg["page_limit"]}


def checklist(paper: Path, pdf_text: str | None) -> dict:
    path = paper / "sections" / "checklist.tex"
    if not path.exists():
        return {"ok": False, "requirement": "Paper checklist", "detail": "sections/checklist.tex is missing"}
    text = path.read_text()
    answers = re.findall(r"\\item\[\] Answer: (\\answer(?:Yes|No|NA)\{\})?", text)
    justs = re.findall(r"\\item\[\] Justification: ?(.*)", text)
    questions = len(re.findall(r"\\item\[\] Question:", text))
    placeholders = len(re.findall(r"TODO|BEGIN INSTRUCTIONS", text))
    in_pdf = pdf_text is None or "NeurIPS Paper Checklist" in pdf_text
    ok = (questions > 0 and len(answers) == questions == len(justs) and all(answers) and all(j.strip() for j in justs)
          and not placeholders and in_pdf and "\\input{sections/checklist}" in (paper / "main.tex").read_text())
    return {"ok": ok, "requirement": "Paper checklist", "questions": questions, "answered": sum(1 for a in answers if a),
            "justified": sum(1 for j in justs if j.strip()), "placeholders": placeholders, "in_pdf": in_pdf}


def fonts(pdf: Path) -> dict:
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
    bad = []
    for line in out:
        m = re.match(r"^(\S+)\s+(.+?)\s+(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s", line)
        if m and (m.group(4) != "yes" or "Type 3" in m.group(2)):
            bad.append(line.strip())
    return {"ok": bool(out) and not bad, "requirement": "Fonts and figures", "fonts": len(out), "not_embedded_or_type3": bad}


def load_config() -> dict:
    """`configs/submission.yaml`, plus `private_identifying_strings` from the untracked `configs/submission.local.yaml`
    (account IDs and other strings that must not be published even as a search list)."""
    cfg = config.load("submission")
    local = config.CONFIGS / "submission.local.yaml"
    if local.exists():
        cfg["private_identifying_strings"] = (yaml.safe_load(local.read_text()) or {}).get("private_identifying_strings", [])
    return cfg


def anonymity(pdf: Path, pdf_text: str, cfg: dict) -> dict:
    meta = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    meta = "\n".join(ln for ln in meta.splitlines() if not ln.startswith(("Creator", "Producer")))   # the TeX engine, not the authors
    hay = (pdf_text + "\n" + meta).lower()
    found = [s for s in cfg["identifying_strings"] if s.lower() in hay]
    private = cfg.get("private_identifying_strings", [])     # from the untracked submission.local.yaml; never written out
    found += [f"<private string {i + 1}>" for i, s in enumerate(private) if s.lower() in hay]
    tex = re.sub(r"(?m)%.*$", "", (pdf.parent / "main.tex").read_text())
    return {"ok": not found and not re.search(r"linenumbers\s*=\s*false|\\nolinenumbers", tex), "requirement": "Anonymous submission",
            "searched_for": cfg["identifying_strings"], "private_strings_searched": len(private), "found": found}


def overflow(paper: Path, cfg: dict) -> dict:
    log = (paper / "main.log").read_text(errors="replace") if (paper / "main.log").exists() else ""
    wide = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    unresolved = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)'.*undefined", log)
    return {"ok": max(wide, default=0.0) <= cfg["overfull_tolerance_pt"] and not unresolved, "requirement": "Fonts and figures",
            "widest_overfull_pt": max(wide, default=0.0), "tolerance_pt": cfg["overfull_tolerance_pt"],
            "unresolved_references": sorted(set(unresolved))}


def inclusion(paper: Path, cfg: dict) -> dict:
    """Every generated table, appendix and required figure is included somewhere, and every table and figure label is cited."""
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", _tex(paper))
    inputs = set(re.findall(r"\\input\{generated/([\w]+)\}", tex))
    files = {p.name for p in (paper / "generated").glob("*.tex")}
    missing = sorted(f for f in files if f.removesuffix(".tex") not in inputs and f not in cfg["not_included"])
    figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{generated/figs/([\w]+)\}", tex))
    missing_figs = sorted(set(cfg["required_figures"]) - figs)
    labels = set(re.findall(r"\\label\{((?:tab|fig):[\w-]+)\}", tex))
    uncited = sorted(lab for lab in labels if not re.search(r"\\ref\{" + re.escape(lab) + r"\}", tex))
    return {"ok": not missing and not missing_figs and not uncited, "requirement": "Paper structure within the submission format",
            "generated_files_not_included": missing, "required_figures_not_included": missing_figs,
            "tables_and_figures_never_cited": uncited}


def run_checks(mode: str, pdf_text: str | None) -> dict:
    cfg, paper = load_config(), config.PAPER
    mock = json.loads((paper / "generated" / "mode.json").read_text())["mockdraft"]
    checks = {"page_limit": page_limit(paper, cfg)}
    if cfg.get("checklist_required", True):
        checks["checklist"] = checklist(paper, pdf_text)
    else:                                       # removed on purpose: it must then really be absent, and the result says so
        present = "\\input{sections/checklist}" in re.sub(r"(?m)%.*$", "", (paper / "main.tex").read_text())
        checks["checklist"] = {"ok": not present, "requirement": "Paper checklist", "required": False,
                               "note": "not required by configs/submission.yaml; the paper is not a complete NeurIPS submission without it"}
    if not mock:
        pdf = paper / "main.pdf"
        checks.update(style=style(paper, cfg), fonts=fonts(pdf), anonymity=anonymity(pdf, pdf_text or "", cfg),
                      overflow=overflow(paper, cfg), inclusion=inclusion(paper, cfg))
    return checks
