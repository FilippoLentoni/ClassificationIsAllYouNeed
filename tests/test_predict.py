"""Templates, predictions, the historical-payload audit and the backup."""
import json
import sqlite3

import numpy as np
import pytest

from ciayn import audit, backup, config, predict, results, templates
from ciayn.backends import exchange
from ciayn.backends.budget import Ledger
from ciayn.backends.cache import CachedBackend
from ciayn.backends.jev import JevBackend
from ciayn.data import datasets
from ciayn.data.serialize import serialize
from conftest import stage_args
from fakes import SECRET, FakeJevServer


def test_template_check_uses_training_examples_only(workspace):
    templates.run(stage_args("mock", "sst2"))
    rec = json.loads(templates.path("mock", "sst2").read_text())
    ds = datasets.load("sst2")
    ids = rec["check"]["ids"]
    assert len(ids) == 20 and set(ids) <= set(ds.ids("train"))
    assert not set(ids) & set(ds.eval_ids) and not set(ids) & set(ds.ids("val"))
    assert rec["instructions"] == ds.task and [c for c, _ in rec["criteria"]] == ds.classes
    assert rec["question_sha256"] == templates.question_sha256(ds)


def test_tabular_template_is_frozen_without_touching_data_rows(live):
    templates.run(stage_args("live", "wdbc"))
    rec = json.loads(templates.path("live", "wdbc").read_text())
    assert rec["check"]["n"] == 0 and live.server.calls == 0
    assert rec["instructions"].endswith("Fields that are not listed are unknown.")


def test_live_template_check_costs_at_most_twenty_requests(live):
    templates.run(stage_args("live", "sst2"))
    assert live.server.calls == 20 == Ledger().count("template")
    train = set(datasets.load("sst2").ids("train"))
    sent = {r["state"] for r in live.server.requests}
    ds = datasets.load("sst2")
    assert sent <= {serialize(ds, i) for i in train}


def test_predict_refuses_without_a_frozen_template_and_after_a_template_change(live, monkeypatch):
    with pytest.raises(RuntimeError, match="no frozen template"):
        predict.run(stage_args("live", "wdbc"))
    templates.run(stage_args("live", "wdbc"))
    ds = datasets.load("wdbc")
    monkeypatch.setattr(ds, "task", ds.task + " Be careful.")
    with pytest.raises(RuntimeError, match="changed after it was frozen"):
        predict.run(stage_args("live", "wdbc"))
    assert live.server.calls == 0


def test_live_predict_saves_every_row_and_checks_the_first_response_alone(live):
    templates.run(stage_args("live", "wdbc"))
    predict.run(stage_args("live", "wdbc"))
    ds = datasets.load("wdbc")
    df = results.read_predictions("live", "wdbc", "jev")
    assert list(df.id) == ds.jev_ids and len(df) == 569 and df.valid.all() and (df["mode"] == "live").all()
    assert np.allclose(df[["p0", "p1"]].sum(axis=1), 1)
    first = exchange.read("predict")[0]
    assert first["ledger_n"] == 1, "the first request is sent on its own"
    s = results.scores("live", ds, "jev", "test", repeat=3)
    assert s.ids == ds.ids("test", 3) and len(s) == 114


def test_invalid_first_response_stops_the_dataset(live):
    templates.run(stage_args("live", "wdbc"))
    live.server.missing_class = True
    with pytest.raises(RuntimeError, match="first wdbc response is invalid"):
        predict.run(stage_args("live", "wdbc"))
    assert live.server.calls == 1


def test_key_never_persisted(live):
    """A live predict run leaves no trace of the key in the cache, the exchange log, results or backup."""
    templates.run(stage_args("live", "wdbc"))
    predict.run(stage_args("live", "wdbc"))
    backup.run()
    files = [p for p in live.root.rglob("*") if p.is_file()]
    assert any(p.suffix == ".sqlite" for p in files) and any(p.suffix == ".parquet" for p in files)
    assert any(p.suffix == ".jsonl" for p in files)
    for p in files:
        data = p.read_bytes()
        assert SECRET.encode() not in data and SECRET[:-4].encode() not in data, f"key leaked into {p}"


def test_mock_predict_writes_both_models_on_the_same_examples(workspace):
    predict.run(stage_args("mock", "german"))
    ds = datasets.load("german")
    for repeat in (0, 7):
        j, x = results.scores("mock", ds, "jev", "test", repeat), results.scores("mock", ds, "xgb", "test", repeat)
        results.same_examples(j, x, "german")
        assert j.ids == ds.ids("test", repeat) and j.modes == ["mock"] == x.modes
    curve = __import__("pandas").read_parquet(results.model_dir("mock", "german", "xgb") / "learning_curve.parquet")
    assert sorted(curve.fraction.unique()) == [0.05, 0.1, 0.25, 0.5, 1.0] and set(curve.repeat) == set(range(10))
    assert not (workspace / "results" / "live").exists()


def test_payload_audit_reports_unlogged_calls_and_adds_nothing_to_the_log(live):
    ds = datasets.load("german")
    xs = [serialize(ds, i) for i in ds.jev_ids[:5]]

    class Stub:                                   # an SDK-level stub: responses are cached but nothing is logged,
        async def __aenter__(self): return self   # like the calls made before request logging existed
        async def __aexit__(self, *a): return None
        async def system_one(self, state, questions):
            from typesafe_sdk import SystemOneResponse
            (name, q), = questions.items()
            probs = FakeJevServer.probabilities(state, list(q.criteria))
            return SystemOneResponse.model_validate({"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0},
                "answers": {name: {"type": "choice", "choice": "good", "confidence": 0.1, "probabilities": probs}}})

    CachedBackend(JevBackend(client_factory=Stub, stage="predict")).predict(ds, xs)
    CachedBackend(JevBackend(stage="predict")).predict(ds, [serialize(ds, ds.jev_ids[10])])   # one logged call
    assert len(exchange.read()) == 1
    audit.run(stage_args("live"))
    res = json.loads((live.root / "results" / "live" / "payload_audit.json").read_text())
    assert res["ledger_entries_without_a_logged_exchange"] == 5
    assert res["cached_responses_without_a_logged_exchange"] == 5
    assert res["regenerated_requests_matching_a_stored_cache_key"] == 5
    assert res["exchange_log_records_created_by_this_audit"] == 0 and len(exchange.read()) == 1
    assert any("smoke" in s for s in res["not_recorded"])


def test_backup_copies_are_identical_and_changes_are_detected(live):
    templates.run(stage_args("live", "wdbc"))
    predict.run(stage_args("live", "wdbc"))
    rec = backup.run()
    assert rec["location"] == str(config.BACKUP) and rec["time"].endswith("Z")
    names = {e["name"] for e in rec["files"]}
    assert names == {"responses.sqlite", "exchanges/predict.jsonl"}
    for e in rec["files"]:
        assert backup.sha256_file(config.BACKUP / e["name"]) == e["sha256"] == backup.sha256_file(config.CACHE / e["name"])
    assert backup.local_matches_latest() == (True, str(config.BACKUP))
    with open(config.CACHE / "exchanges" / "predict.jsonl", "a") as f:
        f.write("{}\n")
    ok, why = backup.local_matches_latest()
    assert not ok and "changed since the latest backup" in why


def test_payload_examples_are_verbatim_log_records(live):
    from ciayn import payloads
    templates.run(stage_args("live", "agnews"))
    ds = datasets.load("agnews")
    order = [ds.eval_ids[i] for i in np.random.default_rng(0).permutation(len(ds.eval_ids))]
    CachedBackend(JevBackend(stage="predict")).predict(ds, [serialize(ds, i) for i in order[:3]])
    rec = payloads.export("agnews")
    logged = exchange.read("predict")
    assert rec["exchange"] in logged, "the exported exchange is identical to its log record"
    assert rec["example_id"] == order[0] and rec["gold_label"] == ds.classes[int(ds.y[order[0]])]
    q = rec["exchange"]["request"]["questions"]["label"]
    assert list(q["criteria"]) == ds.classes and len(q["criteria"]) == 4
    assert len(rec["exchange"]["response"]["answers"]["label"]["probabilities"]) == 4
    assert rec["exchange"]["request"]["state"] == serialize(ds, order[0])


def test_payload_example_is_refused_when_nothing_was_logged(live):
    from ciayn import payloads
    with pytest.raises(RuntimeError, match="no logged exchange"):
        payloads.export("sst2")
