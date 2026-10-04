"""`payload-audit`: what can and cannot be said about Jev calls made before request logging existed.

Those calls have no record in the exchange log and none is created here. For the responses
that are in the cache, the request is regenerated from code and its hash is compared with
the stored cache key: a match shows that the same input and question were sent, but it does
not recover the payload as it was on the wire, and the raw response was never stored.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import config
from .backends import exchange
from .backends.budget import Ledger
from .backends.cache import cache_key
from .backends.jev import JevBackend
from .data import datasets
from .data.serialize import serialize


def audit() -> dict:
    db = sqlite3.connect(config.CACHE / "responses.sqlite")
    cached = {k: mv for k, mv in db.execute("SELECT key, model_version FROM responses WHERE backend = 'jev'")}
    ledger = Ledger().db.execute("SELECT n, stage FROM jev_ledger ORDER BY n").fetchall()
    logged = exchange.read()
    logged_keys = {r["cache_key"] for r in logged if r.get("cache_key")}
    logged_n = {r["ledger_n"] for r in logged if r.get("ledger_n") is not None}
    unlogged_cached = set(cached) - logged_keys
    backend = JevBackend()
    regenerated, per_dataset = set(), {}
    for name in config.DATASETS:
        ds = datasets.load(name)
        qd = backend.question_def(ds)
        keys = {cache_key("jev", backend.model_version, qd, serialize(ds, i)) for i in ds.jev_ids}
        hit = keys & unlogged_cached
        regenerated |= hit
        if hit:
            per_dataset[name] = len(hit)
    unlogged_ledger = [(n, stage) for n, stage in ledger if n not in logged_n]
    by_stage: dict[str, int] = {}
    for _, stage in unlogged_ledger:
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger_entries_without_a_logged_exchange": len(unlogged_ledger),
        "ledger_entries_without_a_logged_exchange_by_stage": by_stage,
        "cached_responses_without_a_logged_exchange": len(unlogged_cached),
        "regenerated_requests_matching_a_stored_cache_key": len(regenerated),
        "regenerated_matches_by_dataset": per_dataset,
        "what_a_match_shows": "the regenerated input text, instructions, label descriptions and model version hash "
                              "to the key under which the response was stored when it was received",
        "not_recorded": ["the request payloads as sent on the wire", "the raw response bodies (only the parsed "
                         "probabilities, chosen label, confidence, token count and latency were stored)",
                         "the smoke-test call, which bypassed the cache: neither its request nor its response was stored"],
        "exchange_log_records_created_by_this_audit": 0,
    }


def run(args) -> int:
    n_before = len(exchange.read())
    res = audit()
    assert len(exchange.read()) == n_before, "the audit must never add to the exchange log"
    out = config.results_dir("live") / "payload_audit.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"payload-audit: {res['ledger_entries_without_a_logged_exchange']} ledger entries have no logged exchange "
          f"{res['ledger_entries_without_a_logged_exchange_by_stage']}; "
          f"{res['regenerated_requests_matching_a_stored_cache_key']} of "
          f"{res['cached_responses_without_a_logged_exchange']} unlogged cached responses are reproduced by a "
          f"regenerated request; written to {out}")
    return 0
