"""Backend factory: the live Jev backend, or a labelled mock predictor in mock mode."""
from __future__ import annotations

from functools import lru_cache

from ..data.datasets import Dataset
from .base import Backend


@lru_cache(maxsize=None)
def mock_model(name: str, model: str = "jev"):
    from ..data import datasets
    from .mock import MockBackend

    return MockBackend(datasets.load(name), model)


def get(backend: str, mode: str, ds: Dataset, stage: str | None = None) -> Backend:
    if backend != "jev":
        raise ValueError(f"unknown backend {backend!r}; Jev is the only model queried from this machine")
    if mode == "mock":
        return mock_model(ds.name)
    from .jev import JevBackend

    return JevBackend(stage=stage)
