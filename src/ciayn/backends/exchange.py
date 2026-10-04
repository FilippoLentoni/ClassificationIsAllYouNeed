"""Request and response log (design D4).

A transport wrapped around the SDK's HTTP transport records the body of every Jev request as
it is sent and the body of every response as it is received. Headers are never recorded, so
the API key cannot enter the log. One JSON line per exchange is appended to
`cache/exchanges/<stage>.jsonl`. Nothing is ever written for an exchange that was not
observed here: there is no back-fill.
"""
from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import httpx2

from .. import config

# Set by the backend around each request: stage, dataset, cache_key, ledger number.
CURRENT: ContextVar[dict | None] = ContextVar("ciayn_exchange", default=None)
KEY_ENV = "TYPESAFE_API_KEY"


def log_dir() -> Path:
    return config.CACHE / "exchanges"


def sdk_version() -> str:
    import typesafe_sdk

    return typesafe_sdk.__version__


def _body(raw: bytes):
    """JSON when it parses (the normal case), otherwise the text as received."""
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


class ExchangeLog:
    def __init__(self, stage: str):
        self.path = log_dir() / f"{stage}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        key = os.environ.get(KEY_ENV, "").strip()
        if key and key in line:        # cannot happen (no headers are recorded); refuse rather than store it
            raise RuntimeError("refusing to write an exchange record that contains the API key")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class LoggingTransport(httpx2.AsyncBaseTransport):
    """Records each request body as sent and each response body as received."""

    def __init__(self, inner: httpx2.AsyncBaseTransport, requested_model: str):
        self.inner, self.requested_model = inner, requested_model

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        ctx = CURRENT.get() or {}
        record = {"ledger_n": ctx.get("ledger_n"), "stage": ctx.get("stage"), "dataset": ctx.get("dataset"),
                  "cache_key": ctx.get("cache_key"), "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "sdk_version": sdk_version(), "requested_model": self.requested_model,
                  "method": request.method, "url": str(request.url).split("?")[0],
                  "request": _body(request.content)}
        t0 = time.perf_counter()
        try:
            response = await self.inner.handle_async_request(request)
            raw = await response.aread()
        except Exception as e:
            record.update(status=None, response=None, error=type(e).__name__, latency_s=time.perf_counter() - t0)
            ExchangeLog(ctx.get("stage") or "unspecified").write(record)
            raise
        body = _body(raw)
        record.update(status=response.status_code, response=body, latency_s=time.perf_counter() - t0,
                      reported_model=body.get("model") if isinstance(body, dict) else None)
        ExchangeLog(ctx.get("stage") or "unspecified").write(record)
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


def read(stage: str | None = None) -> list[dict]:
    files = sorted(log_dir().glob("*.jsonl")) if stage is None else [log_dir() / f"{stage}.jsonl"]
    out = []
    for f in files:
        if f.exists():
            out += [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln]
    return out


def find(cache_keys: set[str], stage: str = "predict") -> dict[str, dict]:
    """Logged records of a stage for the given cache keys (first successful one per key), read as a stream."""
    path, out = log_dir() / f"{stage}.jsonl", {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get("cache_key")
            if k in cache_keys and k not in out and r.get("status") == 200:
                out[k] = r
    return out
