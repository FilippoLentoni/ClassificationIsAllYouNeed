import json

import numpy as np
import pandas as pd
import pytest

from ciayn import config, results
from ciayn.data import datasets
from ciayn.experiments import metrics as m
from conftest import stage_args


def toy3():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])
    return y, P, P.argmax(1)     # predictions 0, 1, 1, 2, 2, 0


def test_three_class_metrics_by_hand():
    y, P, pred = toy3()
    assert m.accuracy(y, pred) == pytest.approx(3 / 6)
    assert m.confusion(y, pred, 3).tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    prec, rec, f1, support = m.per_class(y, pred, 3)
    assert prec.tolist() == [0.5, 0.5, 0.5] and rec.tolist() == [0.5, 0.5, 0.5] and support.tolist() == [2, 2, 2]
    assert m.macro_f1(y, pred, 3) == pytest.approx(0.5)
    # one-vs-rest ROC-AUC: class 0 scores {0.7, 0.4} vs {0.1, 0.3, 0.1, 0.5}: 7 of 8 pairs; class 1: {0.8, 0.3} vs
    # {0.2, 0.5, 0.2, 0.1}: 7 of 8; class 2: {0.7, 0.4} vs {0.1, 0.1, 0.1, 0.4}: 7.5 of 8
    roc, _ = m.auc_scores(y, P)
    assert roc == pytest.approx((7 / 8 + 7 / 8 + 7.5 / 8) / 3)
    assert m.brier(y, P) == pytest.approx(np.mean([0.14, 0.62, 0.06, 0.74, 0.14, 0.62]))


def test_invalid_outputs_are_errors():
    y, P, pred = toy3()
    base = m.all_metrics(y, P, pred)
    pred2, P2 = pred.copy(), P.copy()
    pred2[0], P2[0] = -1, [1 / 3, 1 / 3, 1 / 3]          # a correct item becomes invalid
    bad = m.all_metrics(y, P2, pred2)
    assert bad["accuracy"] < base["accuracy"] and bad["macro_f1"] < base["macro_f1"]
    assert m.confusion(y, pred2, 3).sum() == 5, "an invalid output is in no predicted column"


def test_perfectly_calibrated_predictor_has_ece_near_zero():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p1.size) < p1).astype(int)
    P = np.c_[1 - p1, p1]
    assert m.ece(y, P, P.argmax(1)) < 0.01
    over = P ** 3 / (P ** 3).sum(1, keepdims=True)
    assert m.ece(y, over, over.argmax(1)) > 0.05


def test_fast_scorers_match_sklearn():
    from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p1 = np.round(np.clip(0.3 * y + rng.uniform(0, 0.7, 300), 0, 1), 2)  # rounded: ties
    P = np.c_[1 - p1, p1]
    pred = P.argmax(1)
    pred[:7] = -1
    roc, ap = m.auc_scores(y, P)
    assert roc == pytest.approx(roc_auc_score(y, p1)) and ap == pytest.approx(average_precision_score(y, p1))
    assert m.macro_f1(y, pred, 2) == pytest.approx(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
    assert m.logloss(y, P) == pytest.approx(log_loss(y, np.clip(P, 1e-12, 1), labels=[0, 1]))
    # multiclass, with ties and one class absent from the sample
    y5 = rng.integers(0, 4, 400)
    P5 = np.round(rng.dirichlet(np.ones(5), 400) + 0.3 * np.eye(5)[y5], 2)
    P5 /= P5.sum(1, keepdims=True)
    roc5, ap5 = m.auc_scores(y5, P5)
    onehot = np.eye(5)[y5][:, :4]
    assert roc5 == pytest.approx(roc_auc_score(onehot, P5[:, :4], average="macro"))
    assert ap5 == pytest.approx(average_precision_score(onehot, P5[:, :4], average="macro"))


def test_repeated_split_dataset_reports_mean_and_sd_over_repeats(mock_run):
    m.run(stage_args("mock", "german"))
    res = json.loads((mock_run / "results/mock/german/metrics/metrics.json").read_text())
    ds = datasets.load("german")
    for model in config.MODELS:
        h = res["models"][model]["headline"]["accuracy"]
        assert h["n_repeats"] == 10 and len(h["values"]) == 10 and h["sd"] > 0 and "ci95" not in h
        assert h["value"] == pytest.approx(np.mean(h["values"]))
        assert res["models"][model]["n_evaluated"] == 200
    r3 = results.scores("mock", ds, "jev", "test", 3)
    assert res["models"]["jev"]["headline"]["accuracy"]["values"][3] == pytest.approx(np.mean(r3.pred == r3.y))
    assert res["label_budget"]["jev"] == {"training": 0, "early_stopping_and_selection": 0, "calibration": 0, "threshold_selection": 200}
    assert res["label_budget"]["xgb"]["training"] == 600 and res["label_budget"]["xgb"]["calibration"] == 0
    assert res["label_budget"]["qwen"] == res["label_budget"]["jev"], "Qwen is zero-shot, like Jev"
    assert res["label_budget"]["bert"]["training"] == 600
    assert res["missing_models"] == [] and res["artifact_modes"] == ["mock"]


def test_heldout_dataset_reports_bootstrap_intervals_and_writes_outputs(mock_run):
    m.run(stage_args("mock", "sst2"))
    out = mock_run / "results/mock/sst2/metrics"
    res = json.loads((out / "metrics.json").read_text())
    for model in config.MODELS:
        for name in m.HEADLINE:
            h = res["models"][model]["headline"][name]
            assert h["ci95"][0] <= h["value"] <= h["ci95"][1], (model, name)
        assert res["models"][model]["n_evaluated"] == 872 and len(res["models"][model]["per_class"]) == 2
    assert res["label_budget"]["jev"]["threshold_selection"] == 500 and res["label_budget"]["xgb"]["training"] == 65349
    assert {p.name for p in out.iterdir()} == {"metrics.json", "reliability.pdf", "learning_curve.pdf",
                                               *(f"per_class_{mo}.csv" for mo in config.MODELS)}
    assert set(res["learning_curves"]) == {"xgb", "bert"}, "one learning curve per supervised model"
    for model, curve in res["learning_curves"].items():
        assert [c["fraction"] for c in curve] == [0.05, 0.1, 0.25, 0.5, 1.0] and curve[-1]["n_train"] == 65349
        assert curve[-1]["accuracy"]["value"] == pytest.approx(res["models"][model]["headline"]["accuracy"]["value"])
    assert list(pd.read_csv(out / "per_class_jev.csv").columns) == ["class", "precision", "recall", "f1", "support"]


def test_models_scored_on_different_examples_stop_the_stage(mock_run):
    path = results.pred_path("mock", "wdbc", "xgb")
    df = pd.read_parquet(path)
    df[~((df.repeat == 0) & (df.split == "test") & (df.id == df[(df.repeat == 0) & (df.split == "test")].id.iloc[0]))].to_parquet(path)
    with pytest.raises(RuntimeError, match="missing 1 identifiers"):
        m.run(stage_args("mock", "wdbc"))
    a = results.scores("mock", datasets.load("german"), "jev", "test", 0)
    b = results.scores("mock", datasets.load("german"), "xgb", "test", 1)
    with pytest.raises(RuntimeError, match="not scored on the same examples"):
        results.same_examples(a, b, "german")


@pytest.mark.parametrize("model", ["qwen", "bert"])
def test_any_model_on_different_examples_stops_the_stage(mock_run, model):
    path = results.pred_path("mock", "wdbc", model)
    df = pd.read_parquet(path)
    first = results.scores("mock", datasets.load("wdbc"), "jev", "test", 0).ids[0]
    drop = (df.id == first) if model == "qwen" else ((df.repeat == 0) & (df.split == "test") & (df.id == first))
    df[~drop].to_parquet(path)
    with pytest.raises(RuntimeError, match="missing 1 identifiers"):
        m.run(stage_args("mock", "wdbc"))


def test_thinking_subsample_is_scored_on_its_own_identifiers(mock_run):
    from ciayn.llm import thinking_subsample
    m.run(stage_args("mock", "sst2"))
    res = json.loads((mock_run / "results/mock/sst2/metrics/metrics.json").read_text())
    ds = datasets.load("sst2")
    ids = thinking_subsample(ds)
    sub = res["thinking_subsample"]
    assert sub["n"] == len(ids) == 50 and set(sub["models"]) == {"qwen_think", *config.MODELS}
    s = results.subset(results.scores("mock", ds, "xgb", "test", 0), ids)
    assert sub["models"]["xgb"]["accuracy"] == pytest.approx(np.mean(s.pred == s.y)), "others on the same subsample only"
    assert "qwen_think" not in res["models"], "thinking results never enter the four-model tables"


def test_learning_curve_figure_shows_every_supervised_model(mock_run, monkeypatch):
    seen = []
    real = m.plt.Axes.errorbar
    monkeypatch.setattr(m.plt.Axes, "errorbar", lambda self, *a, **k: seen.append(k.get("label")) or real(self, *a, **k))
    m.run(stage_args("mock", "german"))
    assert seen == ["XGBoost (supervised)", "BERT (supervised)"]
