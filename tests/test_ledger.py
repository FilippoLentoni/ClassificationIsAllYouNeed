"""The Jev inference cap: 60,000 in total, counted in a persistent ledger (design D3)."""
import pytest

from ciayn import config, predict, templates
from ciayn.backends.budget import Budget, BudgetExceeded, DryRunStop, Ledger
from ciayn.backends.jev import JevBackend
from ciayn.data.serialize import serialize
from conftest import stage_args


def test_budgets_config():
    b = config.load("budgets")
    stages = {s: b[s]["jev"] for s in ("smoke", "template", "predict", "explain")}
    assert b["jev_total"] == 60000 and stages == {"smoke": 5, "template": 60, "predict": 13700, "explain": 44000}
    assert sum(stages.values()) <= b["jev_total"]


def test_earlier_inferences_count_toward_the_cap(live, german):
    ledger = Ledger(cap=105)
    for _ in range(99):
        ledger.reserve("predict", 13700)
    ledger.reserve("smoke", 5)                         # the 100 inferences made under the earlier limit
    assert Ledger(cap=105).total() == 100
    with pytest.raises(BudgetExceeded, match="global Jev cap") as e:
        Budget("predict", ledger=Ledger(cap=105)).preflight({"jev": 6})
    assert (e.value.required, e.value.allowed) == (106, 105) and live.server.calls == 0
    Budget("predict", ledger=Ledger(cap=105)).preflight({"jev": 5})


def test_cap_holds_across_separate_invocations(live, german):
    """Each Ledger() is a fresh process view of the same file; requests beyond the cap are never sent."""
    xs = [serialize(german, i) for i in german.ids("test")[:12]]
    for chunk in (xs[:4], xs[4:8], xs[8:]):            # three separate "runs", cap 10
        b = JevBackend(stage="predict", ledger=Ledger(cap=10))
        b._predict(german, chunk)
    assert Ledger().total() == 10 and live.server.calls == 10
    assert any("global Jev cap" in m for m in b.failures)


def test_stage_budget_is_cumulative_across_runs(live, german, monkeypatch):
    monkeypatch.setitem(config.load("budgets"), "smoke", {"jev": 1})
    xs = [serialize(german, i) for i in german.ids("test")[:2]]
    JevBackend(stage="smoke")._predict(german, xs[:1])
    assert JevBackend(stage="smoke")._predict(german, xs[1:]) == [None]
    assert live.server.calls == 1 and Ledger().count("smoke") == 1


def test_stage_over_budget_stops_before_any_call(live, wdbc, monkeypatch):
    templates.run(stage_args("live", "wdbc"))
    monkeypatch.setitem(config.load("budgets"), "predict", {"jev": 500})   # wdbc needs 569
    with pytest.raises(BudgetExceeded) as e:
        predict.run(stage_args("live", "wdbc"))
    assert (e.value.required, e.value.allowed) == (569, 500) and live.server.calls == 0 and Ledger().total() == 0


def test_dry_run_reports_and_makes_no_calls(live, capsys):
    templates.run(stage_args("live", "wdbc"))
    args = stage_args("live", "wdbc")
    args.dry_run = True
    with pytest.raises(DryRunStop):
        predict.run(args)
    out = capsys.readouterr().out
    assert "[dry-run] predict: jev=569 calls; Jev ledger: 0 of 60000 used" in out
    assert live.server.calls == 0 and Ledger().total() == 0


def test_warm_rerun_leaves_ledger_unchanged(live):
    templates.run(stage_args("live", "wdbc"))
    predict.run(stage_args("live", "wdbc"))
    assert Ledger().total() == 569 == Ledger().count("predict") == live.server.calls
    predict.run(stage_args("live", "wdbc"))
    assert live.server.calls == 569 and Ledger().total() == 569
