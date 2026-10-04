"""`escalation-val` stage: Jev on seeded validation items of AG News (change
add-bert-qwen-ensemble, added 2026-10-01 at the user's choice).

AG News has no other Jev predictions on validation items, so the escalation threshold could
not be selected without evaluation labels. This stage asks Jev for 500 items drawn with a fixed seed
from the dataset's validation part (never an evaluation item), with the same cache, ledger and
request log as `predict` and its own stage budget. The predictions are saved apart from the frozen
Jev items, in `results/<mode>/<dataset>/jev/escalation_val.parquet`.
"""
from __future__ import annotations

import numpy as np

from . import backends, config, templates
from .backends.budget import Budget
from .backends.cache import CachedBackend
from .data import datasets
from .data.datasets import Dataset
from .data.serialize import serialize
from .results import write_jev_predictions

STAGE = "escalation_val"


def ids(ds: Dataset) -> list[str]:
    """The seeded validation items for `ds`, or [] for a dataset that already has Jev validation items."""
    cfg = config.load("escalation")
    n = int(cfg["validation_items"].get(ds.name, 0))
    if not n:
        return []
    val = list(ds.repeats[0]["val"])
    pick = np.random.default_rng(int(cfg["seed"])).choice(len(val), min(n, len(val)), replace=False)
    out = [val[i] for i in sorted(pick)]
    if set(out) & set(ds.eval_ids):
        raise RuntimeError(f"{ds.name}: escalation validation items overlap the evaluation items")
    return out


def run(args) -> int:
    names = [n for n in templates.selected(args) if ids(datasets.load(n))]
    budget = Budget(STAGE, dry_run=args.dry_run)
    plan, required = [], 0
    for name in names:
        ds = datasets.load(name)
        backend = backends.get("jev", args.mode, ds, stage=STAGE)
        if args.mode == "live":
            templates.require(args.mode, ds)
        cb = CachedBackend(backend, budget=budget)
        texts = [serialize(ds, i) for i in ids(ds)]
        todo = cb.missing(ds, texts) if backend.remote else []
        required += backend.calls_needed(len(todo))
        plan.append((ds, cb, texts))
        print(f"escalation-val: {name}: {len(texts)} validation items, {len(todo)} not yet cached")
    budget.preflight({"jev": required} if args.mode == "live" else {})
    for ds, cb, texts in plan:
        df = write_jev_predictions(args.mode, ds, list(zip(ids(ds), cb.predict(ds, texts))), name="escalation_val")
        print(f"escalation-val: {ds.name}: {len(df)} predictions saved; invalid {int((~df.valid).sum())}")
    print(f"escalation-val ({args.mode}): remote calls made: {sum(budget.used.values())} {budget.used}")
    return 0
