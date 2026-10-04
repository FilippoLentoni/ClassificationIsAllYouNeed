"""Jev backend: one TypeSafe `Choice` question per example (design D2, D4).

All SDK specifics live here. The API key is read from the environment (the CLI loads
the gitignored `.env` first; exported variables win); it is passed only to the SDK
client and never logged, cached or written to results.

Every request is recorded in the persistent Jev ledger right before it is sent, and SDK
retries are disabled, so each request counts exactly once against the inference cap
(design D3). Every request and response is also written to the exchange log as it crosses
the wire. A failed request returns None: it is not cached and a later run may retry it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np

from .. import config
from ..data.datasets import Dataset
from ..data.serialize import MISSING_NOTE
from . import exchange
from .base import Backend, Prediction
from .budget import BudgetExceeded, Ledger, stage_allowance

log = logging.getLogger(__name__)
KEY_ENV = "TYPESAFE_API_KEY"
GUIDE = "docs/JEV_ACCESS.md"
TRANSPORT_FACTORY = None   # tests replace the network transport under the real SDK client


def api_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{KEY_ENV} is not set. Put it in the project's gitignored .env "
                           f"(or export it); see {GUIDE}. Use --mode mock to run without it.")
    return key


def scrub(msg: str) -> str:
    """Remove the API key from a message, keeping at most its last 4 characters."""
    key = os.environ.get(KEY_ENV, "").strip()
    return msg.replace(key, f"…{key[-4:]}") if key else msg


def instructions(ds: Dataset) -> str:
    return f"{ds.task} {MISSING_NOTE}" if ds.kind == "tabular" else ds.task


def template(ds: Dataset) -> dict:
    """Everything besides the input that determines a request, in the order sent."""
    cfg = config.load("jev")
    return {"dataset": ds.name, "question_name": cfg["question_name"], "instructions": instructions(ds),
            "criteria": [[c, ds.criteria[c]] for c in ds.classes], "model": cfg["model"]}


class JevBackend(Backend):
    name = "jev"
    chunk_size = 200   # cache writes happen per chunk, so interrupted runs resume

    def __init__(self, client_factory=None, model: str | None = None, stage: str | None = None,
                 ledger: Ledger | None = None, transport=None):
        cfg = config.load("jev")
        self.cfg = cfg
        self.model = model or cfg["model"]
        self.stage = stage
        self._client_factory = client_factory     # tests: a stub SDK client (nothing crosses a wire, nothing is logged)
        self._transport = transport               # tests: a fake HTTP transport under the real SDK client
        self._ledger = ledger
        self.last_response_model: str | None = None
        self.failures: list[str] = []

    @property
    def ledger(self) -> Ledger:
        if self._ledger is None:
            self._ledger = Ledger()
        return self._ledger

    @property
    def model_version(self) -> str:
        return self.model

    def question_def(self, ds: Dataset) -> str:
        return json.dumps({"question": self.cfg["question_name"], "instructions": instructions(ds),
                           "criteria": {c: ds.criteria[c] for c in ds.classes}}, sort_keys=True)

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        import httpx2
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        inner = (self._transport if self._transport is not None
                 else TRANSPORT_FACTORY() if TRANSPORT_FACTORY is not None else httpx2.AsyncHTTPTransport())
        # no SDK retries: every HTTP request must be one ledgered, logged inference
        return AsyncTypeSafeClient(api_key=api_key(), model=self.model, timeout=self.cfg["timeout_s"],
                                   retry=RetryPolicy(max_retries=0),
                                   transport=exchange.LoggingTransport(inner, self.model))

    def _question(self, ds: Dataset):
        from typesafe_sdk import Choice

        # criteria keep the dataset's class order
        return Choice(instructions=instructions(ds), criteria={c: ds.criteria[c] for c in ds.classes})

    def to_prediction(self, ds: Dataset, resp, latency: float) -> Prediction:
        """Map a SystemOneResponse to a Prediction in the dataset's class order."""
        ans = resp.choices.get(self.cfg["question_name"])
        tokens = getattr(resp.usage, "input_tokens", None)
        meta = {"backend": self.name, "model_version": self.model, "response_model": resp.model,
                "mode": "live", "latency_s": latency, "input_tokens": tokens,
                "cost_usd": (tokens or 0) * self.cfg["usd_per_mtok_input"] / 1e6}
        if ans is None:
            return Prediction.make(np.ones(len(ds.classes)), valid=False, meta=meta)
        probs = ans.probabilities
        valid = set(probs) == set(ds.classes)
        vec = [float(probs.get(c, 0.0)) for c in ds.classes]
        meta.update(jev_choice=ans.choice, jev_confidence=float(ans.confidence))
        return Prediction.make(vec, valid=valid, meta=meta)

    async def _run(self, ds: Dataset, texts: list[str]) -> list[Prediction | None]:
        from .cache import cache_key

        sem = asyncio.Semaphore(self.cfg["concurrency"])
        question = {self.cfg["question_name"]: self._question(ds)}
        allowed, stage, qd = stage_allowance(self.stage), self.stage or "unspecified", self.question_def(ds)
        async with self._client() as client:
            async def one(text: str) -> Prediction | None:
                async with sem:
                    try:
                        n = self.ledger.reserve(stage, allowed)
                    except BudgetExceeded as e:
                        self.failures.append(str(e))
                        return None
                    token = exchange.CURRENT.set({"ledger_n": n, "stage": stage, "dataset": ds.name,
                                                  "cache_key": cache_key(self.name, self.model, qd, text)})
                    t0 = time.perf_counter()
                    try:
                        resp = await client.system_one(state=text, questions=question)
                    except Exception as e:  # never let the key reach logs or tracebacks
                        msg = f"Jev call failed: {type(e).__name__}: {scrub(str(e))}"
                        log.warning(msg)
                        self.failures.append(msg)
                        return None
                    finally:
                        exchange.CURRENT.reset(token)
                    return self.to_prediction(ds, resp, time.perf_counter() - t0)

            return list(await asyncio.gather(*(one(t) for t in texts)))

    def _predict(self, ds: Dataset, texts: list[str]) -> list[Prediction | None]:
        api_key() if self._client_factory is None else None  # fail fast before any work
        self.failures = []
        preds = asyncio.run(self._run(ds, texts))
        models = {p.meta.get("response_model") for p in preds if p is not None}
        self.last_response_model = next(iter(models)) if len(models) == 1 else ",".join(sorted(map(str, models)))
        return preds


def smoke(args) -> int:
    """One live Choice query on a German Credit example; prints the full distribution."""
    from ..data import datasets
    from ..data.serialize import serialize
    from .budget import Budget

    api_key()
    ds = datasets.load("german")
    ex = ds.ids("test")[0]
    backend = JevBackend(stage="smoke")
    Budget("smoke", dry_run=args.dry_run, ledger=backend.ledger).preflight({"jev": 1})
    pred = backend._predict(ds, [serialize(ds, ex)])[0]  # bypasses the cache on purpose
    if pred is None:
        print(f"error: {'; '.join(backend.failures)}", flush=True)
        return 1
    print(f"requested model: {backend.model}\nresponse model:  {pred.meta.get('response_model')}")
    print(f"example: {ex} (gold: {ds.classes[ds.y[ex]]})")
    for c, p in zip(ds.classes, pred.probs):
        print(f"  P({c}) = {p:.4f}")
    print(f"choice: {pred.meta.get('jev_choice')}  confidence: {pred.meta.get('jev_confidence')}  "
          f"valid: {pred.valid}  input tokens: {pred.meta.get('input_tokens')}  "
          f"latency: {pred.meta['latency_s']:.2f}s")
    if not pred.valid:
        print("error: response did not contain a probability for every class", flush=True)
        return 1
    if pred.meta.get("response_model") != backend.model:
        print(f"error: the API reports model {pred.meta.get('response_model')}, not the pinned {backend.model}; "
              "stop and decide whether to continue on that version", flush=True)
        return 1
    return 0
