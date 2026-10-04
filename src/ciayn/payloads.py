"""`payload-examples`: one complete logged Jev exchange per dataset, copied verbatim from the request log.

The example is the first evaluation example, in a seeded order, whose request is in the log.
Requests made before logging existed are never used: there is no record to copy, and none is
reconstructed.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, templates
from .backends import exchange
from .backends.cache import cache_key
from .backends.jev import JevBackend
from .data import datasets
from .data.serialize import serialize

SEED = 0


def out_path(name: str):
    return config.results_dir("live", "payload_examples") / f"{name}.json"


def export(name: str) -> dict:
    ds = datasets.load(name)
    backend = JevBackend()
    qd = backend.question_def(ds)
    order = [ds.eval_ids[i] for i in np.random.default_rng(SEED).permutation(len(ds.eval_ids))]
    keys = {cache_key("jev", backend.model_version, qd, serialize(ds, i)): i for i in order[:200]}
    found = exchange.find(set(keys))
    hit = next(((k, i) for k, i in keys.items() if k in found), None)
    if hit is None:
        raise RuntimeError(f"{name} has no logged exchange for an evaluation example; nothing can be shown for it")
    key, ex = hit
    rec = {"dataset": name, "example_id": ex, "gold_label": ds.classes[int(ds.y[ex])],
           "selection": f"first evaluation example in a seeded order (seed {SEED}) whose request is in the request log",
           "source": "cache/exchanges/predict.jsonl, copied verbatim", "exchange": found[key]}
    out_path(name).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def run(args) -> int:
    for name in templates.selected(args):
        rec = export(name)
        ex = rec["exchange"]
        ans = next(iter(ex["response"]["answers"].values()))
        print(f"payload-examples: {name}: {rec['example_id']} (ledger #{ex['ledger_n']}), "
              f"{len(next(iter(ex['request']['questions'].values()))['criteria'])} options, "
              f"chose {ans['choice']} (gold {rec['gold_label']}), confidence {ans['confidence']}")
    return 0
