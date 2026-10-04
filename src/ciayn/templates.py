"""`template` stage: check each text dataset's prompt on training examples, then freeze all templates.

A template is everything besides the input that determines a request: the instructions and
the label descriptions in the order sent (design D2). Text datasets are tried on at most 20
training-split examples. The tabular datasets get no check on data rows, because any row can
be an evaluation row in some repeat; `predict` validates their response shape on its first
request instead. `predict` refuses to run in live mode for a dataset without a frozen template.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from . import backends, config
from .backends.budget import Budget
from .backends.cache import CachedBackend
from .backends.jev import template
from .data import datasets
from .data.datasets import Dataset
from .data.serialize import serialize

N_CHECK = 20


def path(mode: str, name: str):
    return config.results_dir(mode, "templates") / f"{name}.json"


def question_sha256(ds: Dataset) -> str:
    return hashlib.sha256(json.dumps(template(ds), sort_keys=True).encode()).hexdigest()


def check_ids(ds: Dataset) -> list[str]:
    """A seeded draw of training-split examples; never an evaluation or validation identifier."""
    train = ds.ids("train")
    rng = np.random.default_rng(config.load("datasets")["seed"])
    return [train[i] for i in sorted(rng.choice(len(train), size=min(N_CHECK, len(train)), replace=False))]


def require(mode: str, ds: Dataset) -> dict:
    p = path(mode, ds.name)
    if not p.exists():
        raise RuntimeError(f"{ds.name} has no frozen template ({p}); run `python -m ciayn template --mode {mode}` first")
    rec = json.loads(p.read_text())
    if rec["question_sha256"] != question_sha256(ds):
        raise RuntimeError(f"the template of {ds.name} changed after it was frozen ({p}); "
                           "re-run the template stage with --refreeze and record why")
    return rec


def selected(args) -> list[str]:
    name = getattr(args, "dataset", "all")
    return list(config.DATASETS) if name in (None, "all") else [name]


def run(args) -> int:
    from .backends import exchange

    budget = Budget("template", dry_run=args.dry_run)
    plan, required = [], 0
    for name in selected(args):
        ds = datasets.load(name)
        cb = CachedBackend(backends.get("jev", args.mode, ds, stage="template"), budget=budget)
        ids = check_ids(ds) if ds.kind == "text" else []
        texts = [serialize(ds, i) for i in ids]
        if cb.backend.remote:
            required += cb.backend.calls_needed(len(cb.missing(ds, texts)))
        plan.append((ds, cb, ids, texts))
    budget.preflight({"jev": required} if args.mode == "live" else {})
    for ds, cb, ids, texts in plan:
        p = path(args.mode, ds.name)
        rec = {**template(ds), "question_sha256": question_sha256(ds), "mode": args.mode,
               "sdk_version": exchange.sdk_version(), "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if p.exists():
            old = json.loads(p.read_text())
            if old["question_sha256"] == rec["question_sha256"]:
                print(f"template: {ds.name} already frozen at {old['frozen_at']}, unchanged")
                continue
            if not getattr(args, "refreeze", False):
                raise RuntimeError(f"the template of {ds.name} differs from the frozen one; pass --refreeze to replace it")
            rec["superseded"] = old.get("superseded", []) + [{k: v for k, v in old.items() if k != "superseded"}]
        if ids:
            preds = cb.predict(ds, texts)
            ok = [p_.valid for p_ in preds]
            rec["check"] = {"split": "train", "n": len(ids), "ids": ids, "valid_responses": int(sum(ok)),
                            "accuracy": float(np.mean([p_.label == ds.y[i] for i, p_ in zip(ids, preds)]))}
            if not all(ok):
                raise RuntimeError(f"template check for {ds.name}: {len(ok) - sum(ok)} of {len(ok)} responses did not "
                                   "contain a probability for every class; the template was not frozen")
        else:
            rec["check"] = {"split": None, "n": 0, "reason": "every row of a repeated-split dataset can be an "
                            "evaluation row; the response shape is validated by the first request of predict"}
        p.write_text(json.dumps(rec, indent=1))
        note = f"checked on {rec['check']['n']} training examples, accuracy {rec['check']['accuracy']:.2f}" if ids else "no data-row check"
        print(f"template: {ds.name} frozen ({len(ds.classes)} options, {note})")
    print(f"template ({args.mode}): remote calls made: {sum(budget.used.values())}")
    return 0
