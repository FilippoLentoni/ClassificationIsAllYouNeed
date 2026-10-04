"""Request and response log: what crossed the wire, and never the key."""
import json

import pytest

from ciayn.backends import exchange
from ciayn.backends.cache import CachedBackend, ResponseCache, cache_key
from ciayn.backends.jev import JevBackend
from ciayn.data.serialize import serialize
from fakes import SECRET


def test_one_record_per_request_with_bodies_as_sent_and_received(live, german):
    xs = [serialize(german, i) for i in german.ids("test")[:4]]
    backend = JevBackend(stage="predict")
    preds = backend._predict(german, xs)
    records = exchange.read("predict")
    assert len(records) == 4 == live.server.calls
    by_state = {r["request"]["state"]: r for r in records}
    for x, p, sent in zip(xs, preds, live.server.requests):
        r = by_state[x]
        assert r["request"] in live.server.requests, "the logged request is the body the server received"
        assert r["request"]["questions"]["label"]["instructions"].startswith("Assess the credit risk")
        assert r["status"] == 200 and r["response"]["answers"]["label"]["probabilities"].keys() == {"good", "bad"}
        assert r["sdk_version"] == exchange.sdk_version() and r["requested_model"] == "jev-1.13.0"
        assert r["reported_model"] == "jev-1.13.0" and r["dataset"] == "german" and r["latency_s"] >= 0
        assert r["cache_key"] == cache_key("jev", backend.model_version, backend.question_def(german), x)
    assert sorted(r["ledger_n"] for r in records) == [1, 2, 3, 4]


def test_failed_request_is_recorded_without_a_response(live, german):
    x = serialize(german, german.ids("test")[0])
    live.server.fail_on = {x}
    assert JevBackend(stage="predict")._predict(german, [x]) == [None]
    (r,) = exchange.read("predict")
    assert r["response"] is None and r["status"] is None and r["error"] == "ConnectError" and r["request"]["state"] == x


def test_log_contains_neither_the_key_nor_headers(live, german):
    xs = [serialize(german, i) for i in german.ids("test")[:3]]
    live.server.fail_on = {xs[1]}                      # the error message carries the key
    CachedBackend(JevBackend(stage="predict"), ResponseCache(live.root / "c.sqlite")).predict(german, xs)
    raw = (exchange.log_dir() / "predict.jsonl").read_text()
    assert SECRET not in raw and SECRET[:-4] not in raw
    assert "authorization" not in raw.lower() and "user-agent" not in raw.lower()
    assert all(set(json.loads(ln)) >= {"request", "response", "ts", "stage"} for ln in raw.splitlines())


def test_a_record_containing_the_key_is_refused(live):
    with pytest.raises(RuntimeError, match="contains the API key"):
        exchange.ExchangeLog("predict").write({"request": {"state": f"leaked {SECRET}"}})
    assert exchange.read("predict") == []
