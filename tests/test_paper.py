"""Paper generator and number check. No LaTeX run here; the build itself is exercised by `make paper-mock`."""
import json

import pytest

from ciayn import backup, config, predict, templates
from ciayn.backends.budget import Ledger
from ciayn.backends.cache import CachedBackend
from ciayn.backends.jev import JevBackend
from ciayn.data import datasets
from ciayn.data.serialize import serialize
from ciayn.experiments import comparison, escalation, explain, importance, metrics, thresholding
from ciayn.paper import build, provenance, verify
from conftest import stage_args

SMALL = ("german", "wdbc", "sst2")


@pytest.fixture
def mock_paper(mock_run, monkeypatch):
    """Every experiment stage run in mock mode on the three small datasets, with fast settings."""
    monkeypatch.setattr(config, "DATASETS", SMALL)
    monkeypatch.setitem(config.load("importance"), "trials", 40)
    for k, v in (("n_examples", 2), ("shap_nsamples", 30), ("lime_samples", 30)):
        monkeypatch.setitem(config.load("explain"), k, v)
    args = stage_args("mock")
    for stage in (metrics, thresholding, importance, explain, comparison, escalation):
        stage.run(args)
    return mock_run


def test_mock_inputs_set_the_watermark_flag_and_everything_is_generated(mock_paper):
    g = build.generate("mock")
    assert "\\def\\mockdraft{}" in (g.out / "mockdraft.tex").read_text() and build.is_mock_draft(g)
    assert json.loads((g.out / "mode.json").read_text())["artifact_modes"] == ["mock"]
    macros = (g.out / "macros.tex").read_text()
    for name in ("AccuracyJevGerman", "AccuracyXgbSst", "EceJevWdbc", "ThresholdValidationJevWdbc", "IsVarRatioSstNOneZeroZero",
                 "ExDropShapGermanKThree", "PriorAccDeusserSst", "JevCap"):
        assert f"\\newcommand{{\\{name}}}" in macros, name
    assert "\\newcommand{\\JevCap}{60{,}000}" in macros
    for name in ("tab_metrics", "tab_calibration", "tab_labels", "tab_thresholds", "tab_importance", "tab_explain", "tab_prior",
                 "tab_datasets", "tab_cost", "appendix_payloads", "appendix_datasets", "appendix_perclass"):
        assert (g.out / f"{name}.tex").read_text().strip(), name
    for fig in ("reliability", "learning_curves", "importance", "faithfulness", "threshold_wdbc_jev", "attributions_german"):
        assert (g.out / "figs" / f"{fig}.pdf").exists(), fig
    labels = (g.out / "tab_labels.tex").read_text()
    assert "Jev, zero-shot & 0 & 0 & 0 & 200" in labels and "XGBoost, supervised & 600 & 200 & 0 & 200" in labels
    assert "No logged exchange exists in mock mode" in (g.out / "appendix_payloads.tex").read_text()
    for name in ("tab_costs", "tab_escalation", "tab_thinking", "appendix_qwen"):
        assert (g.out / f"{name}.tex").read_text().strip(), name
    for fig in ("f1_by_labels", "escalation"):
        assert (g.out / "figs" / f"{fig}.pdf").exists(), fig
    assert "Qwen3.6-27B, zero-shot & 0 & 0 & 0 & 200 & 200" in labels, "escalation-threshold labels for the zero-shot models"
    assert "BERT, supervised & 600 & 200 & 0 & 200 & --" in labels
    for name in ("EscFOneGerman", "ThinkSubFOneQwenThinkSst", "AwsCostUSDBert", "CostPerKJevGerman"):
        assert f"\\newcommand{{\\{name}}}" in macros, name
    sources = json.loads((g.out / "macro_sources.json").read_text())
    assert sources["AccuracyJevGerman"]["source"] == "results/mock/german/metrics/metrics.json" or \
        sources["AccuracyJevGerman"]["source"].endswith("results/mock/german/metrics/metrics.json")


def test_a_prior_number_without_a_check_date_is_refused(mock_paper, monkeypatch):
    prior = config.load("prior_work")
    monkeypatch.setitem(prior["sources"][0], "checked_on", None)
    with pytest.raises(RuntimeError, match="has no check date"):
        build.generate("mock")


def test_every_dataset_needs_a_prior_entry_or_an_explicit_none_found(mock_paper, monkeypatch):
    monkeypatch.setitem(config.load("prior_work"), "none_found", [])
    with pytest.raises(RuntimeError, match="says nothing about \\['german'\\]"):
        build.generate("mock")
    assert "German Credit & none found" not in ""   # the row exists only when the search result is recorded


def test_live_appendix_fails_without_a_logged_exchange(workspace, monkeypatch):
    g = build.Generated("live", workspace / "generated")
    (g.out / "figs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="has no exported logged exchange"):
        build.payload_appendix(g)


def test_a_mock_artifact_among_live_results_makes_a_draft(workspace):
    d = config.results_dir("live", "german", "metrics")
    (d / "metrics.json").write_text(json.dumps({"mode": "live", "artifact_modes": ["live", "mock"]}))
    g = build.Generated("live", workspace / "generated")
    assert not build.is_mock_draft(g)
    g.read("german", "metrics", "metrics")
    assert build.is_mock_draft(g)


def test_reported_usage_covers_every_request_in_the_ledger(live):
    templates.run(stage_args("live", "sst2"))                      # 20 template requests
    ds = datasets.load("german")
    xs = [serialize(ds, i) for i in ds.jev_ids[:7]]
    live.server.fail_on = {xs[3]}
    CachedBackend(JevBackend(stage="predict")).predict(ds, xs)     # 7 requests, one of which fails
    u = provenance.jev_usage()
    assert u["requests_in_ledger"] == Ledger().total() == 27
    assert u["by_stage"]["template"]["requests"] == 20 and u["by_stage"]["predict"]["requests"] == 7
    assert u["failed_requests"] == 1 and u["requests_in_log"] == 27
    assert u["requests_with_token_counts"] == 26 and u["requests_without_token_counts"] == 1
    assert u["input_tokens"] == 26 * 100 and u["usd"] == pytest.approx(26 * 100 * 0.042 / 1e6)
    g = build.Generated("live", live.root / "generated")
    (g.out / "figs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="run `python -m ciayn provenance --mode live`"):
        build.provenance_part(g)                                   # the paper reads saved usage; it does not recompute it
    provenance.run(stage_args("live"))
    build.provenance_part(g)
    assert g.macros["JevInferencesUsed"] == "27" and g.macros["JevFailedRequests"] == "1"
    assert g.sources["JevInferencesUsed"].endswith("provenance.json")


def test_verify_detects_a_number_that_differs_from_the_saved_outputs(mock_paper):
    build.generate("mock")
    ok = verify.verify("mock", check_pdf=False)
    assert ok["generated_files_match_saved_outputs"] and ok["macros_defined"] > 100
    assert all(r["source"] for r in ok["macros"])
    path = config.PAPER / "generated" / "macros.tex"
    path.write_text(path.read_text().replace("\\newcommand{\\JevCap}{60{,}000}", "\\newcommand{\\JevCap}{70{,}000}"))
    bad = verify.verify("mock", check_pdf=False)
    assert bad["generated_files_that_differ"] == ["macros.tex"]
    with pytest.raises(RuntimeError, match="generated files differ from the saved outputs: macros.tex"):
        verify.run(stage_args("mock"))


def test_pdf_text_matching_tolerates_line_breaks_and_hyphenation():
    page = "accuracy of 0.960 [0.947,\n0.972] on the offi-\ncial split; model jev-\n1.13.0 used 13,682 requests"
    assert verify.in_text("0.960", page) and verify.in_text("13{,}682", page) and verify.in_text("jev-1.13.0", page)
    assert verify.in_text("official", page) and not verify.in_text("0.961", page)
    assert verify.plain(r"agentcore\_role 5\%") == "agentcore_role 5%"


def test_macro_names_are_letters_only():
    assert build.macro_name("IsVarRatio", "Sst", "n", 100) == "IsVarRatioSstNOneZeroZero"
    assert build.macro_name("macro_f1", "Jev", "German") == "MacroFOneJevGerman"


# ---- NeurIPS submission format checks --------------------------------------------------------------

def _paper(tmp_path, monkeypatch, main_page=8, checklist_answer="\\answerYes{}", options=""):
    from ciayn.paper import submission
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "generated").mkdir()
    monkeypatch.setattr(config, "PAPER", paper)
    (paper / "neurips_2026.sty").write_text("style")
    import hashlib
    (paper / "STYLE_SOURCE.json").write_text(json.dumps({"sha256": hashlib.sha256(b"style").hexdigest()}))
    (paper / "main.tex").write_text(f"\\documentclass{{article}}\n\\usepackage{options}{{neurips_2026}}\n\\input{{generated/tab_a}}\n"
                                    "\\label{tab:a} Table~\\ref{tab:a}\n\\input{sections/checklist}\n")
    (paper / "main.aux").write_text(f"\\newlabel{{end:main}}{{{{12}}{{{main_page}}}{{Conclusion}}{{section.12}}{{}}}}\n")
    (paper / "generated" / "tab_a.tex").write_text("x")
    (paper / "sections" / "checklist.tex").write_text(
        "\\item[] Question: Q1?\n\\item[] Answer: " + checklist_answer + "\n\\item[] Justification: because.\n")
    return submission, paper


def test_format_checks_pass_on_a_conforming_paper(tmp_path, monkeypatch):
    sub, paper = _paper(tmp_path, monkeypatch)
    cfg = config.load("submission")
    assert sub.page_limit(paper, cfg) == {"ok": True, "requirement": "Page limit", "main_text_ends_on_page": 8, "limit": 9}
    assert sub.checklist(paper, "NeurIPS Paper Checklist")["ok"] and sub.style(paper, cfg)["ok"]
    assert sub.inclusion(paper, cfg | {"required_figures": []})["ok"]


def test_each_format_violation_is_named(tmp_path, monkeypatch):
    sub, paper = _paper(tmp_path, monkeypatch, main_page=10, checklist_answer="\\answerTODO{}", options="[preprint]")
    cfg = config.load("submission")
    assert not sub.page_limit(paper, cfg)["ok"] and sub.page_limit(paper, cfg)["requirement"] == "Page limit"
    c = sub.checklist(paper, "NeurIPS Paper Checklist")
    assert not c["ok"] and c["answered"] == 0 and c["placeholders"] == 1 and c["requirement"] == "Paper checklist"
    s = sub.style(paper, cfg)
    assert not s["ok"] and s["options"] == ["preprint"], "a non-anonymous style option is refused"
    (paper / "neurips_2026.sty").write_text("edited")
    assert not sub.style(paper, cfg)["sha256_matches_downloaded_file"]
    (paper / "generated" / "tab_b.tex").write_text("y")
    inc = sub.inclusion(paper, cfg | {"required_figures": ["reliability"]})
    assert inc["generated_files_not_included"] == ["tab_b.tex"] and inc["required_figures_not_included"] == ["reliability"]
    main = (paper / "main.tex").read_text().replace("Table~\\ref{tab:a}", "")
    (paper / "main.tex").write_text(main)
    assert sub.inclusion(paper, cfg | {"required_figures": []})["tables_and_figures_never_cited"] == ["tab:a"]


def test_overflow_and_identifying_strings(tmp_path, monkeypatch):
    sub, paper = _paper(tmp_path, monkeypatch)
    cfg = config.load("submission")
    (paper / "main.log").write_text("Overfull \\hbox (22.1pt too wide) in paragraph\nLaTeX Warning: Reference `tab:x' on page 3 undefined")
    o = sub.overflow(paper, cfg)
    assert not o["ok"] and o["widest_overfull_pt"] == 22.1 and o["unresolved_references"] == ["tab:x"]
    monkeypatch.setattr(sub.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(stdout="Title: x\nAuthor: \n"))
    pdf = paper / "main.pdf"
    assert sub.anonymity(pdf, "clean text", cfg)["ok"]
    cfg = {**cfg, "private_identifying_strings": ["123456789012"]}
    bad = sub.anonymity(pdf, "weights in s3://ciayn-baseline-123456789012-us-east-2", cfg)
    assert not bad["ok"] and set(bad["found"]) == {"<private string 1>", "ciayn-baseline"}
    assert "123456789012" not in json.dumps(bad)


def test_verification_fails_and_names_the_requirement(mock_paper, monkeypatch, tmp_path):
    from ciayn.paper import submission
    build.generate("mock")
    (config.PAPER / "main.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(stdout=""))
    monkeypatch.setattr(verify, "in_text", lambda v, t: True)
    monkeypatch.setattr(submission, "run_checks", lambda mode, text: {"page_limit": {"ok": False, "requirement": "Page limit",
                                                                                     "main_text_ends_on_page": 11, "limit": 9}})
    with pytest.raises(RuntimeError, match='requirement "Page limit"'):
        verify.run(stage_args("mock", no_build=False))
