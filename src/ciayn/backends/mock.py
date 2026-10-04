"""Labelled synthetic stand-ins for Jev and the supervised baseline (mock mode).

No model is trained. For an example, the logits are

    frac * (margin * onehot(gold) + item_noise) + sum of seeded per-unit effects

where a unit is a `name: value` line (tabular) or a word (text), `frac` is the share of the
example's units present in the input, and the margin grows with log(K - 1) so that many-class
datasets stay plausible. A full input is recognised by its text. A masked input
(explanations) cannot be, so the caller names the example first with `set_parent`. A
temperature turns the logits into probabilities. Outputs are deterministic and independent
of batching or order, and every output has meta.mode = "mock".

"mock-xgb" is addressed by example id (`predict_ids`), per split repeat, and its margin
shrinks with the training fraction so that a mock learning curve rises.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .. import config
from ..data.datasets import Dataset
from ..data.serialize import serialize, units
from .base import Backend, Prediction


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("\x00".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


class MockBackend(Backend):
    remote = False
    cacheable = False
    mode = "mock"

    def __init__(self, ds: Dataset, model: str = "jev", seed: int | None = None):
        cfg = config.load("mock")
        self.p = cfg[model]
        self.model = model
        self.name = f"mock-{model}"
        self.seed = cfg["seed"] if seed is None else seed
        self.ds = ds
        self.k = len(ds.classes)
        self.margin = self.p["margin"] + np.log(max(self.k - 1, 1))
        self._index: dict[str, str] | None = None
        self.parent: str | None = None
        self._effects: dict[str, np.ndarray] = {}

    @property
    def model_version(self) -> str:
        return f"{self.name}-seed{self.seed}"

    # ---- the synthetic model -------------------------------------------------------------------

    def _item_signal(self, ex_id: str, repeat: int | str = "") -> np.ndarray:
        noise = _rng(self.seed, self.model, "item", repeat, ex_id).normal(0, self.p["noise"], self.k)
        return self.margin * np.eye(self.k)[int(self.ds.y[ex_id])] + noise

    def _effect(self, unit: str, n_units: int) -> np.ndarray:
        if unit not in self._effects:
            self._effects[unit] = _rng(self.seed, self.model, "unit", unit).normal(0, 1.0, self.k)
        return self._effects[unit] * self.p["feature_noise"] / np.sqrt(n_units)

    def _probs(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.p["temperature"]
        q = np.exp(z - z.max())
        return q / q.sum()

    def _prediction(self, q: np.ndarray, text_len: int) -> Prediction:
        meta = {"backend": self.name, "model_version": self.model_version, "mode": "mock",
                "latency_s": self.p["latency_s"], "cost_usd": text_len / 4 * self.p["usd_per_mtok"] / 1e6,
                "jev_confidence": round(float((self.k * q.max() - 1) / (self.k - 1)), 2)}
        return Prediction.make(q, meta=meta)

    # ---- by text (mock-jev) --------------------------------------------------------------------

    def _lookup(self, text: str) -> str | None:
        if self._index is None:
            if self.ds.kind == "tabular":
                self._index = {serialize(self.ds, i): i for i in self.ds.frame.index}
            else:
                self._index = {}
                for i, t in zip(self.ds.frame.index, self.ds.frame["text"]):
                    self._index.setdefault(t, i)
        return self._index.get(text)

    def set_parent(self, ex_id: str | None) -> None:
        """Name the example whose masked versions are about to be predicted."""
        self.parent = ex_id

    def logits(self, text: str) -> np.ndarray:
        ex = self._lookup(text) or self.parent
        if ex is None:
            raise ValueError("mock: unknown input; call set_parent(example id) before predicting masked inputs")
        all_units = serialize(self.ds, ex).splitlines() if self.ds.kind == "tabular" else units(self.ds, ex)
        present = text.splitlines() if self.ds.kind == "tabular" else text.split()
        frac = len(present) / max(len(all_units), 1)
        effects = sum((self._effect(u, len(all_units)) for u in present), np.zeros(self.k))
        return frac * self._item_signal(ex) + effects

    def _predict(self, ds: Dataset, texts: list[str]) -> list[Prediction]:
        if ds.name != self.ds.name:
            raise ValueError(f"{self.name} was built for {self.ds.name}, not {ds.name}")
        return [self._prediction(self._probs(self.logits(t)), len(t)) for t in texts]

    # ---- by id (mock-xgb) ----------------------------------------------------------------------

    def predict_ids(self, ids: list[str], repeat: int = 0, fraction: float = 1.0) -> np.ndarray:
        """Probabilities for example ids, as a supervised model trained on `fraction` of a repeat's train part."""
        scale = 0.35 + 0.65 * fraction ** 0.3
        out = []
        for ex in ids:
            signal = self._item_signal(ex, repeat)
            gold = np.eye(self.k)[int(self.ds.y[ex])]
            out.append(self._probs(signal - (1 - scale) * self.margin * gold))
        return np.array(out)
