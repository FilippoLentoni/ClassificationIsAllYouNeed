"""SQLite response cache keyed on sha256(backend, model_version, question_def, input_text)."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .. import config
from ..data.datasets import Dataset
from .base import Backend, Prediction
from .budget import Budget


def cache_key(backend: str, model_version: str, question_def: str, text: str) -> str:
    h = hashlib.sha256()
    for part in (backend, model_version, question_def, text):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class ResponseCache:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.CACHE / "responses.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, backend TEXT, "
                        "model_version TEXT, payload TEXT NOT NULL)")

    def get_many(self, keys: list[str]) -> dict[str, Prediction]:
        out: dict[str, Prediction] = {}
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 500):
            chunk = uniq[i:i + 500]
            q = f"SELECT key, payload FROM responses WHERE key IN ({','.join('?' * len(chunk))})"
            out.update((k, Prediction.from_json(p)) for k, p in self.db.execute(q, chunk))
        return out

    def put_many(self, rows: list[tuple[str, str, str, Prediction]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                            [(k, b, v, p.to_json()) for k, b, v, p in rows])
        self.db.commit()


class CachedBackend:
    """Wraps a backend: serves hits from the cache, calls the backend only for misses."""

    def __init__(self, backend: Backend, cache: ResponseCache | None = None, budget: Budget | None = None):
        self.backend = backend
        self.cache = cache or ResponseCache()
        self.budget = budget

    def keys(self, ds: Dataset, texts: list[str]) -> list[str]:
        b, qd = self.backend, self.backend.question_def(ds)
        return [cache_key(b.name, b.model_version, qd, t) for t in texts]

    def missing(self, ds: Dataset, texts: list[str]) -> list[str]:
        """Distinct uncached inputs."""
        if not self.backend.cacheable:
            return []
        keys = self.keys(ds, texts)
        hits = self.cache.get_many(keys)
        return list(dict.fromkeys(t for k, t in zip(keys, texts) if k not in hits))

    def predict(self, ds: Dataset, texts: list[str]) -> list[Prediction]:
        if not self.backend.cacheable:
            return self.backend._predict(ds, texts)
        keys = self.keys(ds, texts)
        hits = self.cache.get_many(keys)
        todo = list(dict.fromkeys((k, t) for k, t in zip(keys, texts) if k not in hits))
        step = getattr(self.backend, "chunk_size", None) or len(todo) or 1
        for i in range(0, len(todo), step):  # persist per chunk so interrupted runs resume
            chunk = todo[i:i + step]
            if self.budget is not None:
                self.budget.charge(self.backend, len(chunk))
            preds = self.backend._predict(ds, [t for _, t in chunk])
            b = self.backend
            done = [(k, p) for (k, _), p in zip(chunk, preds) if p is not None]  # failures are not cached
            self.cache.put_many([(k, b.name, b.model_version, p) for k, p in done])
            hits.update(done)
        return [hits.get(k) or self.missing_prediction(ds) for k in keys]

    def missing_prediction(self, ds: Dataset) -> Prediction:
        """Stand-in for an item without a response: invalid, uniform, counted as an error."""
        return Prediction.make([1.0] * len(ds.classes), valid=False,
                               meta={"backend": self.backend.name, "model_version": self.backend.model_version,
                                     "mode": self.backend.mode, "missing": True})
