"""The stages added for BERT, Qwen, the comparison and the escalation never make a Jev request:
Jev results come from the saved predictions (experiment-harness: Jev ledger unchanged)."""
import importlib

import pytest

from conftest import stage_args

STAGES = ["ciayn.experiments.metrics", "ciayn.experiments.thresholding", "ciayn.experiments.comparison",
          "ciayn.experiments.escalation"]


@pytest.mark.parametrize("module", STAGES)
def test_stage_leaves_the_jev_ledger_unchanged(mock_run, module, monkeypatch):
    from ciayn import backends
    from ciayn.backends.budget import Ledger

    ledger = Ledger()
    for _ in range(3):
        ledger.reserve("predict", 10)
    before = Ledger().total()
    monkeypatch.setattr(backends, "get", lambda *a, **k: pytest.fail("a Jev backend was requested"))
    from ciayn.experiments import metrics
    for name in ("german", "wdbc", "sst2"):          # the datasets the mock fixture holds
        metrics.run(stage_args("mock", name))       # comparison reads the metrics outputs
        importlib.import_module(module).run(stage_args("mock", name))
    assert Ledger().total() == before == 3


@pytest.mark.parametrize("module", ["ciayn.encoder.cli", "ciayn.llm.cli"])
def test_bert_and_qwen_fetch_leave_the_jev_ledger_unchanged(mock_run, module, monkeypatch):
    import argparse
    from ciayn import backends
    from ciayn.backends.budget import Ledger
    Ledger().reserve("predict", 10)
    monkeypatch.setattr(backends, "get", lambda *a, **k: pytest.fail("a Jev backend was requested"))
    importlib.import_module(module).fetch(argparse.Namespace(mode="mock", dataset="german", dry_run=False))
    assert Ledger().total() == 1
