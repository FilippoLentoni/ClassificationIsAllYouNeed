"""Jev backend: the real SDK client over a fake server, so nothing leaves the machine."""
import numpy as np
import pytest

from ciayn import config
from ciayn.backends import exchange, jev
from ciayn.backends.budget import Ledger
from ciayn.backends.cache import CachedBackend, ResponseCache
from ciayn.backends.jev import JevBackend
from ciayn.data import datasets
from ciayn.data.serialize import serialize
from fakes import SECRET, FakeJevServer


def test_mapping_follows_class_order(live, german):
    texts = ["a", "bb", "ccc"]
    preds = JevBackend(stage="predict")._predict(german, texts)
    for t, p in zip(texts, preds):
        expected = list(FakeJevServer.probabilities(t, german.classes).values())
        assert np.allclose(p.probs, expected), "probabilities are mapped by label, not by response order"
        assert p.valid and p.label == int(np.argmax(p.probs))
        assert p.meta["response_model"] == "jev-1.13.0" and p.meta["input_tokens"] == 100 and p.meta["mode"] == "live"
    assert [list(r["questions"]["label"]["criteria"]) for r in live.server.requests] == [german.classes] * 3


def test_multiclass_request_carries_every_option_in_class_order(live):
    ds = datasets.load("agnews")
    p = JevBackend(stage="predict")._predict(ds, [serialize(ds, ds.eval_ids[0])])[0]
    q = live.server.requests[0]["questions"]["label"]
    assert list(q["criteria"]) == ds.classes and len(p.probs) == 4 and abs(p.probs.sum() - 1) < 1e-6
    assert q["criteria"]["Sci/Tech"] == ds.criteria["Sci/Tech"]
    assert q["instructions"] == ds.task, "text datasets get no missing-field note"


def test_missing_class_is_invalid(live, german):
    live.server.missing_class = True
    p = JevBackend(stage="predict")._predict(german, ["x"])[0]
    assert not p.valid and p.label == -1


def test_missing_key_fails_before_any_request(live, monkeypatch, german, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JEV_ACCESS.md"):
        JevBackend(stage="predict")._predict(german, ["x"])
    from ciayn.cli import main
    monkeypatch.setattr(config, "ROOT", config.ROOT / "nonexistent")  # no .env to load
    assert main(["smoke", "--mode", "live"]) == 2
    assert "docs/JEV_ACCESS.md" in capsys.readouterr().err
    assert live.server.calls == 0 and Ledger().total() == 0


def test_key_loaded_from_dotenv_and_export_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"TYPESAFE_API_KEY={SECRET}\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    config.load_env()
    assert jev.api_key() == SECRET
    monkeypatch.setenv("TYPESAFE_API_KEY", "exported_value")
    config.load_env()
    assert jev.api_key() == "exported_value"


def test_scrub_keeps_last_four(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    assert SECRET not in jev.scrub(f"401 for key {SECRET}")
    assert jev.scrub(f"k={SECRET}").endswith(SECRET[-4:])


def test_sdk_retries_disabled_and_transport_is_logged(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    import typesafe_sdk
    seen = {}
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", lambda **kw: seen.update(kw))
    JevBackend()._client()
    assert seen["retry"].max_retries == 0
    assert isinstance(seen["transport"], exchange.LoggingTransport)


def test_failed_request_is_counted_not_cached_and_retried_later(live, german):
    xs = [serialize(german, i) for i in german.ids("test")[:5]]
    live.server.fail_on = {xs[2]}
    cache = ResponseCache(live.root / "c.sqlite")
    preds = CachedBackend(JevBackend(stage="predict"), cache).predict(german, xs)
    assert Ledger().total() == 5, "the failed request counts against the cap"
    assert not preds[2].valid and preds[2].meta["missing"] and all(p.valid for i, p in enumerate(preds) if i != 2)
    assert all(SECRET not in str(p.meta) for p in preds)
    live.server.fail_on = set()
    again = CachedBackend(JevBackend(stage="predict"), cache).predict(german, xs)  # only the failed item is retried
    assert Ledger().total() == 6 and all(p.valid for p in again)


def test_smoke_prints_the_distribution_and_flags_a_different_model(live, capsys):
    from ciayn.cli import main
    assert main(["smoke", "--mode", "live"]) == 0
    out = capsys.readouterr().out
    assert "P(good) = " in out and "P(bad) = " in out and "response model:  jev-1.13.0" in out and SECRET not in out
    assert Ledger().count("smoke") == 1
    live.server.model = "jev-2.0.0"
    assert main(["smoke", "--mode", "live"]) == 1
    assert "not the pinned" in capsys.readouterr().out
