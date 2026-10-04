"""Uniform probabilistic prediction interface (design D4)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..data.datasets import Dataset

INVALID = -1  # label of an output that could not be mapped to a class


@dataclass
class Prediction:
    probs: np.ndarray               # over the dataset's ordered class set, sums to 1
    label: int                      # argmax, or INVALID
    valid: bool = True
    meta: dict = field(default_factory=dict)

    @classmethod
    def make(cls, probs, valid: bool = True, meta: dict | None = None) -> "Prediction":
        p = np.clip(np.asarray(probs, dtype=float), 0.0, None)
        if not valid or not np.isfinite(p).all() or p.sum() <= 0:
            p, valid = np.full(len(p), 1.0 / len(p)), False
        p = p / p.sum()
        return cls(p, int(np.argmax(p)) if valid else INVALID, valid, dict(meta or {}))

    def to_json(self) -> str:
        return json.dumps({"probs": self.probs.tolist(), "label": self.label, "valid": self.valid, "meta": self.meta})

    @classmethod
    def from_json(cls, s: str) -> "Prediction":
        d = json.loads(s)
        return cls(np.asarray(d["probs"], dtype=float), d["label"], d["valid"], d["meta"])


class Backend(ABC):
    """A model returning class distributions for serialized inputs.

    Subclasses implement `_predict`, which receives only cache misses.
    """

    name: str = "base"
    remote: bool = True   # whether calls cost money / count against budgets
    cacheable: bool = True  # local, deterministic backends skip the response cache
    mode: str = "live"

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    def question_def(self, ds: Dataset) -> str:
        """Everything besides the input that determines the response (part of the cache key)."""
        return json.dumps({"task": ds.task, "classes": ds.classes, "criteria": ds.criteria}, sort_keys=True)

    def calls_needed(self, n_inputs: int) -> int:
        """Remote calls required to predict `n_inputs` uncached inputs."""
        return n_inputs if self.remote else 0

    @abstractmethod
    def _predict(self, ds: Dataset, texts: list[str]) -> list[Prediction]: ...
