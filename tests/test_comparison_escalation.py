"""Comparison of F1, cost and latency, and the Jev->Qwen escalation, from saved predictions only."""
import json

import numpy as np
import pandas as pd
import pytest
from conftest import stage_args

from ciayn import config, results
from ciayn.data import datasets
from ciayn.experiments import comparison as c
from ciayn.experiments import escalation as e
from ciayn.experiments import metrics as m


def _write_run(model, ds, run):
    p = results.model_dir("mock", ds, model) / "run.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), **run}))


def test_every_cost_traces_to_recorded_seconds_or_tokens(mock_run):
    m.run(stage_args("mock", "sst2"))
    price = config.load("baseline")["price_usd_per_hour"]
    _write_run("qwen", "sst2", {"region": "us-east-2", "load_s": 300.0,
                                "jobs": {"Infer": {"instance_type": "ml.g6e.12xlarge", "cost_usd": 10.0, "billable_seconds": 2744.0}},
                                "timing": {"nothink": {"items": 1372, "seconds": 686.0}, "latency_s": [0.2, 0.4, 0.3]},
                                "run_totals": {"nothink_items": 13620, "nothink_seconds": 1800.0, "think_seconds": 200.0,
                                               "latency_seconds": 30.0}})
    _write_run("bert", "sst2", {"region": "us-east-2", "instance_type": "ml.g5.xlarge", "cost_usd": 1.2, "infer_s_per_item": 0.002})
    p = results.model_dir("mock", "sst2", "xgb") / "run.json"       # a run that recorded no inference time
    p.write_text(json.dumps({k: v for k, v in json.loads(p.read_text()).items() if k != "infer_s_per_item"}))
    res = c.run_dataset(datasets.load("sst2"), "mock")
    q = res["costs"]["qwen"]
    assert q["usd_per_1k_inference_only"] == pytest.approx(13.12 / 3600 * 0.5 * 1000)
    assert q["usd_per_1k"] == pytest.approx((10.0 - 13.12 / 3600 * 230.0) * (686.0 / 1800.0) / 1372 * 1000), "load time is in the whole-job figure"
    assert q["usd_per_1k"] > q["usd_per_1k_inference_only"]
    assert q["latency"]["throughput_items_per_s"] == pytest.approx(2.0) and q["latency"]["single_request_n"] == 3
    assert q["latency"]["single_request_mean_s"] == pytest.approx(0.3)
    b = res["costs"]["bert"]
    assert b["usd_per_1k"] == pytest.approx(price["ml.g5.xlarge"] / 3600 * 0.002 * 1000) and b["training_usd"] == 1.2
    jev = results.read_predictions("mock", "sst2", "jev")
    assert res["costs"]["jev"]["usd_per_1k"] == pytest.approx(jev.cost_usd.mean() * 1000)
    assert res["costs"]["xgb"]["usd_per_1k"] is None, "a time that was not recorded is not estimated"
    curves = res["f1_by_labels"]
    assert curves["jev"][0]["n_train"] == 0 and curves["qwen"][0]["n_train"] == 0 and len(curves["bert"]) == 5
    assert (mock_run / "results/mock/sst2/comparison/f1_by_labels.pdf").exists()


def test_escalation_endpoints_unreachable_errors_and_selected_point(mock_run):
    ds = datasets.load("german")
    res = e.run_dataset(ds, "mock")
    for r, rep in enumerate(res["repeats"]):
        j, q = results.scores("mock", ds, "jev", "test", r), results.scores("mock", ds, "qwen", "test", r)
        assert rep["curve"][0]["share_escalated"] == 0 and rep["curve"][0]["accuracy"] == pytest.approx(np.mean(j.pred == j.y))
        assert rep["curve"][-1]["share_escalated"] == 1 and rep["curve"][-1]["accuracy"] == pytest.approx(np.mean(q.pred == q.y))
        assert rep["at"] in rep["curve"] and rep["n_validation_labels"] == 200
        u = rep["unreachable"]
        top = j.conf >= j.conf.max() - 1e-12
        assert u["errors_at_max"] == int(((j.pred != j.y) & top).sum())
    assert res["threshold_selected"] and res["summary"]["threshold"]["n_repeats"] == 10


def test_threshold_does_not_depend_on_evaluation_labels(mock_run):
    ds = datasets.load("wdbc")
    before = [r["threshold"] for r in e.run_dataset(ds, "mock")["repeats"]]
    test0 = set(ds.ids("test", 0)) - set(ds.ids("val", 0))
    for model in ("jev", "qwen"):
        p = results.pred_path("mock", "wdbc", model)
        df = pd.read_parquet(p)
        rows = df.id.isin(test0)
        df.loc[rows, "y"] = 1 - df.loc[rows, "y"]
        df.to_parquet(p)
    after = e.run_dataset(ds, "mock")["repeats"][0]["threshold"]
    assert after == before[0]


def test_no_validation_items_means_no_selected_threshold():
    from ciayn.backends.base import INVALID  # noqa: F401
    y = np.array([0, 1, 1, 0])
    P = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.6, 0.4]])
    s = results.Scores(["a", "b", "c", "d"], y, P, P.argmax(1), np.ones(4, bool), ["mock"])
    ts = e.thresholds(s.conf)
    assert ts[0] == 0 and ts[-1] == e.ALL
    best = e.select([{"t": 0.5, "share_escalated": 0.5, "macro_f1": 0.8}, {"t": 0.9, "share_escalated": 0.9, "macro_f1": 0.8},
                     {"t": 0.0, "share_escalated": 0.0, "macro_f1": 0.7}])
    assert best["t"] == 0.5, "ties go to fewer escalations"
