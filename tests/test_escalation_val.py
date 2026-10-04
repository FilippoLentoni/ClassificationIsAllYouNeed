"""Jev on seeded validation items of AG News, for the escalation threshold only."""
import pytest
from conftest import stage_args

from ciayn import config, escalation_val
from ciayn.backends.budget import BudgetExceeded, Ledger
from ciayn.data import datasets


@pytest.mark.parametrize("name", ["agnews"])
def test_items_come_from_the_validation_part_never_the_evaluation_items(name):
    ds = datasets.load(name)
    ids = escalation_val.ids(ds)
    assert len(ids) == 500 and set(ids) <= set(ds.repeats[0]["val"]) and not set(ids) & set(ds.eval_ids)
    assert ids == escalation_val.ids(ds), "seeded"
    assert not set(ids) & set(ds.jev_ids), "kept apart from the frozen Jev items"


def test_other_datasets_need_none():
    for n in ("german", "wdbc", "sst2"):
        assert escalation_val.ids(datasets.load(n)) == []


def test_stage_budget_is_enforced_before_any_request(live, monkeypatch):
    import shutil
    from pathlib import Path
    shutil.copytree(Path(__file__).resolve().parents[1] / "results" / "live" / "templates", config.results_dir("live") / "templates")
    monkeypatch.setitem(config.load("budgets"), "escalation_val", {"jev": 3})
    with pytest.raises(BudgetExceeded):
        escalation_val.run(stage_args("live", "agnews"))
    assert live.server.calls == 0 and Ledger().total() == 0


def test_mock_run_writes_the_validation_predictions_apart(workspace):
    escalation_val.run(stage_args("mock", "agnews"))
    import pandas as pd
    df = pd.read_parquet(config.RESULTS / "mock" / "agnews" / "jev" / "escalation_val.parquet")
    assert sorted(df.id) == sorted(escalation_val.ids(datasets.load("agnews")))
    assert not (config.RESULTS / "mock" / "agnews" / "jev" / "predictions.parquet").exists()
