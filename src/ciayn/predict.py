"""`predict` stage: Jev predictions for every example it is scored or tuned on (design D1, D3).

Jev predicts all rows of the tabular datasets, the complete held-out split of the text
datasets, and SST-2's threshold-selection items. In live mode a dataset must have a frozen
template, and the first request of a dataset is sent alone so that a wrong response shape
stops the stage before the rest is spent. In mock mode the synthetic baseline's predictions
are written too, so that the whole pipeline runs offline.
"""
from __future__ import annotations

import logging

from . import backends, templates
from .backends.budget import Budget
from .backends.cache import CachedBackend
from .data import datasets
from .data.serialize import serialize
from .results import write_jev_predictions

log = logging.getLogger(__name__)


def run(args) -> int:
    budget = Budget("predict", dry_run=args.dry_run)
    plan, required = [], 0
    for name in templates.selected(args):
        ds = datasets.load(name)
        backend = backends.get("jev", args.mode, ds, stage="predict")
        if args.mode == "live":
            templates.require(args.mode, ds)
        cb = CachedBackend(backend, budget=budget)
        texts = [serialize(ds, i) for i in ds.jev_ids]
        todo = cb.missing(ds, texts) if backend.remote else []
        required += backend.calls_needed(len(todo))
        plan.append((ds, backend, cb, texts, todo))
        if backend.remote:
            print(f"predict: {ds.name}: {len(texts)} examples, {len(todo)} not yet cached")
    budget.preflight({"jev": required} if args.mode == "live" else {})  # dry-run report: calls needed, ledger, cap

    for ds, backend, cb, texts, todo in plan:
        if todo:   # one request first: a wrong response shape must not cost the whole dataset
            first = cb.predict(ds, todo[:1])[0]
            if not first.valid:
                raise RuntimeError(f"predict: the first {ds.name} response is invalid or missing "
                                   f"({'; '.join(getattr(backend, 'failures', [])) or 'no probability for every class'}); "
                                   "stopped before sending the rest")
        preds = cb.predict(ds, texts)
        df = write_jev_predictions(args.mode, ds, list(zip(ds.jev_ids, preds)))
        ev = df.set_index("id").loc[ds.eval_ids]
        log.info("%s/jev (%s): %d predictions, accuracy on evaluation examples %.3f", ds.name, backend.name, len(df),
                 (ev.label == ev.y).mean())
        print(f"predict: {ds.name}: {len(df)} predictions saved; invalid {int((~df.valid).sum())}, "
              f"without a response {int(df.missing.sum())}")
        for msg in sorted(set(getattr(backend, "failures", [])))[:5]:
            print(f"  {msg}")
        if args.mode == "mock":
            from .baseline.mockfit import write_mock_baseline

            write_mock_baseline(ds)
    print(f"predict ({args.mode}): remote calls made: {sum(budget.used.values())} {budget.used}")
    return 0
