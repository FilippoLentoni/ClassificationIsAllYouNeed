import numpy as np
import pytest

from ciayn.backends.base import INVALID, Prediction
from ciayn.backends.budget import Budget, BudgetExceeded, DryRunStop
from ciayn.backends.cache import CachedBackend, ResponseCache
from ciayn.data.serialize import serialize
from fakes import CountingBackend


def texts(ds, n=40):
    return [serialize(ds, i) for i in ds.ids("test")[:n]]


def test_warm_rerun_makes_zero_calls(german, tmp_path):
    fake = CountingBackend()
    cb = CachedBackend(fake, ResponseCache(tmp_path / "c.sqlite"))
    xs = texts(german)
    cold = cb.predict(german, xs)
    assert fake.calls == len(set(xs))
    # new cache object on the same file = a fresh process
    cb2 = CachedBackend(fake, ResponseCache(tmp_path / "c.sqlite"))
    warm = cb2.predict(german, xs)
    assert fake.calls == len(set(xs)), "warm rerun must not call the backend"
    for a, b in zip(cold, warm):
        assert np.array_equal(a.probs, b.probs) and a.label == b.label and a.meta == b.meta


def test_partial_cache_only_calls_misses(german, tmp_path):
    fake = CountingBackend()
    cb = CachedBackend(fake, ResponseCache(tmp_path / "c.sqlite"))
    xs = texts(german, 20)
    cb.predict(german, xs[:10])
    cb.predict(german, xs)
    assert fake.inputs == 20


def test_probability_invariants(german, tmp_path):
    cb = CachedBackend(CountingBackend(), ResponseCache(tmp_path / "c.sqlite"))
    for p in cb.predict(german, texts(german)):
        assert p.probs.shape == (2,)
        assert abs(p.probs.sum() - 1) < 1e-6
        assert p.label == int(np.argmax(p.probs))


def test_invalid_prediction_is_uniform_and_unlabelled():
    p = Prediction.make([0, 0, 0, 0], valid=True)
    assert not p.valid and p.label == INVALID and np.allclose(p.probs, 0.25)
    q = Prediction.make([0.2, 0.8], valid=False)
    assert q.label == INVALID and np.allclose(q.probs, 0.5)


def test_budget_exceeded_raises_before_any_call(german, tmp_path):
    fake = CountingBackend(batch=10)
    budget = Budget("predict", {"fake": 3})
    cb = CachedBackend(fake, ResponseCache(tmp_path / "c.sqlite"), budget)
    xs = texts(german, 40)  # needs 4 calls
    with pytest.raises(BudgetExceeded) as e:
        budget.preflight({"fake": fake.calls_needed(len(cb.missing(german, xs)))})
    assert (e.value.required, e.value.allowed) == (4, 3)
    assert "4" in str(e.value) and "3" in str(e.value)
    with pytest.raises(BudgetExceeded):
        cb.predict(german, xs)
    assert fake.calls == 0


def test_budget_within_limit_and_dry_run(german, tmp_path):
    fake = CountingBackend(batch=10)
    cb = CachedBackend(fake, ResponseCache(tmp_path / "c.sqlite"), Budget("predict", {"fake": 4}))
    cb.predict(german, texts(german, 40))
    assert fake.calls == 4
    dry = Budget("predict", {"fake": 100}, dry_run=True)
    with pytest.raises(DryRunStop):
        dry.preflight({"fake": 7})
