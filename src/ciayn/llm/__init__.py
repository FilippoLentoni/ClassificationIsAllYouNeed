"""Self-hosted Qwen3.6-27B, zero-shot, on SageMaker (change add-bert-qwen-ensemble, design D3-D5)."""
from __future__ import annotations

import numpy as np

from .. import config
from ..data.datasets import Dataset


def thinking_subsample(ds: Dataset) -> list[str]:
    """The seeded evaluation items (repeat 0) that Qwen also answers with thinking enabled."""
    cfg = config.load("llm")["thinking"]
    ids = list(ds.ids("test", 0))
    n = min(int(cfg["subsample_per_dataset"]), len(ids))
    pick = np.random.default_rng(int(cfg["seed"])).choice(len(ids), n, replace=False)
    return [ids[i] for i in sorted(pick)]


def latency_sample(ds: Dataset) -> list[str]:
    """The seeded evaluation items sent one at a time to measure single-request latency."""
    cfg = config.load("llm")["latency"]
    ids = list(ds.ids("test", 0))
    n = min(int(cfg["sample_per_dataset"]), len(ids))
    pick = np.random.default_rng(int(cfg["seed"]) + 1).choice(len(ids), n, replace=False)
    return [ids[i] for i in sorted(pick)]
