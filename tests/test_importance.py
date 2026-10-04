import json

import numpy as np
import pytest

from ciayn.experiments import importance as imp
from conftest import stage_args


def pool(n=600, seed=0, k=3):
    rng = np.random.default_rng(seed)
    p_max = rng.uniform(1 / k + 0.05, 1.0, n)
    y = rng.integers(0, k, n)
    correct = rng.random(n) < p_max                   # a calibrated model: the loss tracks 1 - p_max
    pred = np.where(correct, y, (y + 1) % k)
    P = np.full((n, k), 0.0)
    P[np.arange(n), pred] = p_max
    P[np.arange(n), (pred + 1) % k] = 1 - p_max
    pred[:5] = -1                                     # a few invalid outputs: always errors
    return P, pred, y


def test_every_sampling_probability_is_positive_and_they_sum_to_one():
    P, *_ = pool()
    P[:10] = np.eye(3)[0]                             # items predicted with probability 1: zero predicted uncertainty
    q = imp.proposal(P, alpha=1.0, eps=0.1)
    assert np.all(q > 0) and abs(q.sum() - 1) < 1e-12 and q[:10].min() >= 0.1 / len(q) - 1e-15
    certain = np.tile(np.eye(3)[0], (50, 1))          # no uncertainty anywhere: falls back to uniform
    assert np.allclose(imp.proposal(certain, 1.0, 0.1), 1 / 50)


def test_only_drawn_labels_are_revealed_and_the_proposal_ignores_labels():
    P, pred, y = pool()
    oracle = imp.LabelOracle(y)
    q = imp.proposal(P, 1.0, 0.1)
    rng = np.random.default_rng(3)
    imp.estimate(P, pred, oracle, q, n=10, trials=7, rng=rng)
    assert oracle.calls == 70 and len(oracle.revealed) <= 70 < len(y), "labels are requested for the drawn items only"
    drawn = set(np.random.default_rng(3).choice(len(q), size=(7, 10), p=q).ravel().tolist())
    assert oracle.revealed == drawn
    # the proposal is a function of the predictions: permuting the labels cannot change it
    assert np.array_equal(q, imp.proposal(P, 1.0, 0.1))
    shuffled = imp.LabelOracle(np.random.default_rng(0).permutation(y))
    e = imp.estimate(P, pred, shuffled, q, 10, 7, np.random.default_rng(3))
    assert shuffled.revealed == drawn and e["accuracy"].shape == (7,)


def test_estimator_is_unbiased_and_beats_uniform_when_uncertainty_tracks_loss():
    P, pred, y = pool()
    truth = np.mean(pred == y)
    q = imp.proposal(P, 1.0, 0.1)
    e = imp.estimate(P, pred, imp.LabelOracle(y), q, n=20, trials=5000, rng=np.random.default_rng(0))
    assert abs(e["accuracy"].mean() - truth) < 4 * e["accuracy"].std() / np.sqrt(5000)
    u = imp.estimate(P, pred, imp.LabelOracle(y), np.full(len(q), 1 / len(q)), 20, 5000, np.random.default_rng(1))
    assert abs(u["accuracy"].mean() - truth) < 4 * u["accuracy"].std() / np.sqrt(5000)
    assert e["accuracy"].var() < u["accuracy"].var()
    # unbiased for any positive q, even a badly chosen one
    bad = np.random.default_rng(5).dirichlet(np.ones(len(q)))
    b = imp.estimate(P, pred, imp.LabelOracle(y), bad, 20, 20000, np.random.default_rng(2))
    assert abs(b["accuracy"].mean() - truth) < 4 * b["accuracy"].std() / np.sqrt(20000)


def test_stage_estimates_accuracy_only_on_the_fixed_pool(mock_run):
    imp.run(stage_args("mock", "german"))
    imp.run(stage_args("mock", "sst2"))
    out = mock_run / "results/mock/german/importance"
    res = json.loads((out / "importance.json").read_text())
    assert res["pool"]["n"] == 1000 and res["estimated_quantity"] == "accuracy" and res["budgets"] == [25, 50, 100, 200, 400]
    assert set(res["samplers"]) == {"uniform", "importance"} and res["q"]["min"] > 0 and res["q"]["sum"] == pytest.approx(1)
    assert set(res["samplers"]["uniform"]["400"]) == {"mean", "bias", "variance", "rmse", "coverage95"}
    assert res["label_oracle"]["labels_requested"] == res["label_oracle"]["draws"]
    assert abs(res["samplers"]["importance"]["400"]["bias"]) < 0.01 and res["trials"] == 1000
    assert {p.name for p in out.iterdir()} == {"importance.json", "rmse_vs_budget.pdf"}
    assert json.loads((mock_run / "results/mock/sst2/importance/importance.json").read_text())["pool"]["n"] == 872
    assert "macro" not in json.dumps(res).lower()
