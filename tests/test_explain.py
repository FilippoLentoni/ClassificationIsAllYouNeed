import json

import numpy as np
import pytest

from ciayn import backends, config
from ciayn.backends import exchange
from ciayn.backends.budget import BudgetExceeded, Ledger
from ciayn.backends.cache import CachedBackend
from ciayn.data import datasets
from ciayn.data.serialize import units
from ciayn.experiments import explain as ex
from conftest import stage_args


def fn(ds, ex_id):
    return ex.ProbabilityFunction(CachedBackend(backends.get("jev", "mock", ds)), ds, ex_id)


def test_kernel_shap_attributes_every_feature_and_is_additive(wdbc):
    f = fn(wdbc, wdbc.eval_ids[0])
    r = ex.kernel_shap(f, nsamples=200)
    assert r["attributions"].shape == (30,) and np.all(np.isfinite(r["attributions"]))
    assert r["additivity_error"] < 1e-6, "KernelSHAP is exactly additive with a single background"
    assert r["calls"] <= 200 + 2


def test_lime_gives_every_word_a_weight(sst2):
    cfg = config.load("explain")
    ex_id = ex.choose_examples(sst2, cfg)[0][0]
    f = fn(sst2, ex_id)
    r = ex.lime_explain(f, n=200, width=25, seed=0)
    assert r["attributions"].shape == (len(units(sst2, ex_id)),) and np.count_nonzero(r["attributions"]) == f.F
    assert r["calls"] == 200


def test_deletion_of_top_features_beats_random(wdbc):
    f = fn(wdbc, wdbc.eval_ids[0])
    phi = ex.kernel_shap(f, 200)["attributions"]
    drop_top, _ = ex.deletion(f, np.argsort(-phi), 3)
    rng = np.random.default_rng(0)
    assert drop_top > np.mean([ex.deletion(f, rng.permutation(f.F), 3)[0] for _ in range(20)])


def test_text_examples_follow_the_selection_rule():
    cfg = config.load("explain")
    for name in ("sst2", "agnews"):
        ds = datasets.load(name)
        ids, rule = ex.choose_examples(ds, cfg)
        assert len(ids) == 20 and set(ids) <= set(ds.eval_ids) and ids == ex.choose_examples(ds, cfg)[0]
        assert all(6 <= len(units(ds, i)) <= 40 for i in ids) and rule["eligible"] <= rule["evaluation_examples"]
    ids, rule = ex.choose_examples(datasets.load("german"), cfg)
    assert rule["rule"] == "all rows" and len(ids) == 20


def test_request_bound_per_example():
    assert ex.requests_per_example(config.load("explain")) == 423


@pytest.fixture
def small(monkeypatch):
    cfg = config.load("explain")
    for k, v in (("n_examples", 2), ("shap_nsamples", 30), ("lime_samples", 30)):
        monkeypatch.setitem(cfg, k, v)
    return cfg


def test_stage_stops_before_any_request_when_the_worst_case_exceeds_the_budget(live, small, monkeypatch):
    monkeypatch.setitem(config.load("budgets"), "explain", {"jev": 100})     # 2 examples need up to 2 * 83
    with pytest.raises(BudgetExceeded) as e:
        ex.run(stage_args("live", "wdbc"))
    assert (e.value.required, e.value.allowed) == (2 * ex.requests_per_example(small), 100)
    assert live.server.calls == 0 and Ledger().total() == 0


def test_live_explanations_are_ledgered_logged_and_live_labelled(live, small):
    ex.run(stage_args("live", "wdbc"))
    res = json.loads((live.root / "results/live/wdbc/explain/explain.json").read_text())
    assert res["mode"] == "live" and res["artifact_modes"] == ["live"] and res["n_examples"] == 2
    assert res["jev_requests"] == live.server.calls == Ledger().count("explain") <= 2 * ex.requests_per_example(small)
    assert len(exchange.read("explain")) == live.server.calls and res["cost_usd"]["logged_requests"] == live.server.calls
    assert res["cost_usd"]["input_tokens"] == 100 * live.server.calls
    assert set(res["faithfulness"]) == {"kernel_shap", "lime"} and set(res["faithfulness"]["lime"]) == {"1", "3", "5"}
    assert res["shap_additivity_error"]["max"] < 1e-6 and res["evaluations_per_explanation"]["lime"] == 30
    e0 = res["examples"][0]
    assert len(e0["kernel_shap"]["attributions"]) == 30 == len(e0["units"]) and e0["units"] == datasets.load("wdbc").features
    calls = live.server.calls
    ex.run(stage_args("live", "wdbc"))                     # a rerun is served from the cache
    assert live.server.calls == calls
    again = json.loads((live.root / "results/live/wdbc/explain/explain.json").read_text())
    assert again["jev_requests"] == calls and again["jev_requests_this_run"] == 0, "the recorded total comes from the request log"
    assert again["positive_given_empty_input"] is not None and 0 <= again["positive_given_empty_input"] <= 1
    assert not (live.root / "results" / "mock").exists()


def test_mock_stage_writes_tables_and_figures(mock_run, small):
    ex.run(stage_args("mock", "sst2"))
    out = mock_run / "results/mock/sst2/explain"
    res = json.loads((out / "explain.json").read_text())
    assert res["mode"] == "mock" and res["unit"] == "word" and res["jev_requests"] == 0
    assert {p.name for p in out.iterdir()} == {"explain.json", "attributions.pdf", "faithfulness.pdf"}
    assert -1 <= res["agreement"]["kendall_tau_mean"] <= 1 and 0 <= res["agreement"]["top3_overlap_mean"] <= 1
