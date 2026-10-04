import numpy as np
import pytest

from ciayn import backends
from ciayn.backends.mock import MockBackend
from ciayn.data import datasets
from ciayn.data.serialize import serialize, units


@pytest.mark.parametrize("name", ["german", "sst2", "agnews"])
def test_mock_jev_deterministic_order_independent_and_labelled(name):
    ds = datasets.load(name)
    texts = [serialize(ds, i) for i in ds.eval_ids[:100]]
    a = MockBackend(ds)._predict(ds, texts)
    b = MockBackend(ds)._predict(ds, texts[::-1])[::-1]
    for x, y in zip(a, b):
        assert np.array_equal(x.probs, y.probs) and x.label == y.label
    assert all(p.meta["mode"] == "mock" and p.meta["backend"] == "mock-jev" for p in a)
    assert all(abs(p.probs.sum() - 1) < 1e-6 and len(p.probs) == len(ds.classes) for p in a)
    other = MockBackend(ds, seed=1)._predict(ds, texts)
    assert any(not np.array_equal(x.probs, y.probs) for x, y in zip(a, other))


@pytest.mark.parametrize("name", ["german", "wdbc", "sst2", "agnews"])
def test_mock_is_plausible_not_perfect_without_key_or_aws(name, monkeypatch):
    for var in ("TYPESAFE_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    ds = datasets.load(name)
    ids = ds.eval_ids[:400]
    preds = backends.get("jev", "mock", ds)._predict(ds, [serialize(ds, i) for i in ids])
    acc = np.mean([p.label == ds.y[i] for i, p in zip(ids, preds)])
    assert 0.5 < acc < 0.97


def test_masked_inputs_need_a_parent_and_change_the_output(sst2):
    mock = MockBackend(sst2)
    ex = next(i for i in sst2.eval_ids if len(units(sst2, i)) >= 8)
    with pytest.raises(ValueError, match="set_parent"):
        mock._predict(sst2, [serialize(sst2, ex, drop={0, 1})])
    mock.set_parent(ex)
    full, masked, empty = mock._predict(sst2, [serialize(sst2, ex), serialize(sst2, ex, drop={0, 1}),
                                               serialize(sst2, ex, drop=range(len(units(sst2, ex))))])
    assert not np.array_equal(full.probs, masked.probs)
    assert np.allclose(empty.probs, 0.5), "with every word removed there is no evidence"


def test_mock_baseline_is_deterministic_and_its_learning_curve_rises(german):
    mock = MockBackend(german, "xgb")
    ids = german.ids("test", 2)
    full = mock.predict_ids(ids, repeat=2)
    assert np.array_equal(full, MockBackend(german, "xgb").predict_ids(ids, repeat=2))
    assert not np.array_equal(full, mock.predict_ids(ids, repeat=3)), "a different fit in every repeat"
    y = german.labels(ids)
    acc = [np.mean(mock.predict_ids(ids, 2, f).argmax(1) == y) for f in (0.05, 0.25, 1.0)]
    assert acc[0] < acc[2] and acc[1] <= acc[2]
