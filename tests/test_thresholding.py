import numpy as np
import pytest

from ciayn import config
from ciayn.experiments import thresholding as t

C = np.array([[0.0, 1.0], [5.0, 0.0]])  # German Credit: missing a bad applicant costs 5


def test_german_cost_matrix_is_canonical_and_analytic_threshold():
    assert t.cost_matrix("german", ["good", "bad"]).tolist() == C.tolist()
    assert not t.cost_is_illustrative("german") and t.cost_is_illustrative("wdbc") and t.cost_is_illustrative("sst2")
    assert t.analytic_threshold(C) == pytest.approx(1 / 6)


def test_rule_is_greater_or_equal_at_the_boundary():
    p = np.array([0.29, 0.30, 0.31])
    assert t.decide(p, 0.30).tolist() == [0, 1, 1], "P(positive) exactly equal to t is predicted positive"
    assert t.decide(p, 0.0).tolist() == [1, 1, 1] and t.decide(np.array([1.0]), 1.0).tolist() == [1]


def test_metrics_at_threshold_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    r = t.at_threshold(y, p, 0.5, C)          # decisions 1,0,1,0,0 -> TP=1 FN=1 FP=1 TN=2
    assert (r["precision"], r["recall"], r["accuracy"]) == (0.5, 0.5, 0.6)
    assert r["f1_positive"] == pytest.approx(0.5) and r["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert r["mean_cost"] == pytest.approx((5 + 1) / 5)
    # an invalid output is the wrong decision for its row, whatever its probability
    bad = t.at_threshold(y, p, 0.5, C, valid=np.array([False, True, True, True, True]))
    assert bad["recall"] == 0.0 and bad["mean_cost"] == pytest.approx((5 + 5 + 1) / 5)


def calibrated(n, seed):
    rng = np.random.default_rng(seed)
    p1 = rng.beta(0.6, 1.4, n)
    return (rng.random(n) < p1).astype(int), p1


def test_validation_selected_threshold_is_deterministic_and_near_analytic_when_calibrated():
    y, p = calibrated(20_000, 2)
    a, b = t.select_threshold(y, p, C), t.select_threshold(y, p, C)
    assert a == b and abs(a - t.analytic_threshold(C)) < 0.05


def test_tie_break_prefers_the_threshold_closest_to_analytic_then_smaller():
    y, p = np.array([0, 1]), np.array([0.2, 0.8])   # every t in (0.2, 0.8] has zero cost
    assert t.select_threshold(y, p, C) == pytest.approx(0.21)
    sym = np.array([[0.0, 1.0], [1.0, 0.0]])        # analytic 0.5 lies inside the zero-cost interval
    assert t.select_threshold(y, p, sym) == pytest.approx(0.5)


def test_selection_uses_validation_labels_only():
    yv, pv = calibrated(500, 3)
    yt, pt = calibrated(500, 4)
    a = t.three_thresholds(yv, pv, yt, pt, C)
    b = t.three_thresholds(yv, pv, np.random.default_rng(0).permutation(yt), pt, C)
    assert a["thresholds"] == b["thresholds"], "test labels must not influence any threshold"
    assert a["n_validation_labels"] == 500
    assert set(a["at"]) == {"default", "analytic", "validation"}
    assert a["abs_diff_validation_vs_analytic"] == pytest.approx(abs(a["thresholds"]["validation"] - 1 / 6))


def test_stage_reports_every_repeat_and_writes_a_figure_per_model(mock_run):
    import json
    from conftest import stage_args
    t.run(stage_args("mock"))                      # the two non-binary datasets are skipped
    for name, n_val, illustrative, repeats in (("german", 200, False, 10), ("wdbc", 114, True, 10), ("sst2", 500, True, 1)):
        out = mock_run / "results/mock" / name / "thresholding"
        res = json.loads((out / "thresholding.json").read_text())
        assert res["cost_matrix"]["illustrative"] is illustrative and res["n_repeats"] == repeats
        assert {p.name for p in out.iterdir()} == {"thresholding.json", *(f"threshold_sweep_{mo}.pdf" for mo in config.MODELS)}
        assert list(res["models"]) == list(config.MODELS)
        jev = res["models"]["jev"]
        assert jev["n_validation_labels"] == n_val and len(jev["repeats"]) == repeats
        s = jev["summary"]["at"]["validation"]["mean_cost"]
        assert s["n_repeats"] == repeats and s["value"] == pytest.approx(np.mean([r["at"]["validation"]["mean_cost"] for r in jev["repeats"]]))
        assert set(jev["summary"]["at"]) == {"default", "analytic", "validation"}
        for point in ("default", "validation"):
            c = jev["summary"]["counts"][point]
            assert sum(c.values()) == repeats * jev["n_test"], "every test example of every repeat is in exactly one cell"
            cost = res["cost_matrix"]["rows_true_cols_pred"]
            pooled = (cost[1][0] * c["fn"] + cost[0][1] * c["fp"]) / sum(c.values())
            assert pooled == pytest.approx(jev["summary"]["at"][point]["mean_cost"]["value"]), "counts reproduce the mean cost"
    assert res["models"]["xgb"]["n_validation_labels"] == 2000, "the baseline selects on its full validation part"
    assert json.loads((mock_run / "results/mock/sst2/thresholding/thresholding.json").read_text())["models"]["jev"]["summary"]["thresholds"]["analytic"]["value"] == 0.5
    assert not (mock_run / "results/mock/agnews").exists()
