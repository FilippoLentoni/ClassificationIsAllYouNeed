"""Project paths and YAML config loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("CIAYN_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"          # response cache, ledger and exchange log
DATA = ROOT / "cache"           # downloaded datasets (kept apart so tests can redirect CACHE alone)
PAPER = ROOT / "paper"
LOGS = ROOT / "logs"
SPLITS = ROOT / "results" / "splits"   # frozen evaluation identifiers and their manifest
DOCS = ROOT / "docs"
BACKUP = ROOT.parent / f"{ROOT.name}_backup" / "cache-latest"   # used until the project bucket exists
LEDGER = CACHE / "responses.sqlite"   # the persistent Jev request ledger lives in the response cache

DATASETS = ("german", "wdbc", "sst2", "agnews")   # Banking77 was removed on 2026-10-01 (results/live/removed_datasets.json)
BINARY = ("german", "wdbc", "sst2")
MODELS = ("jev", "qwen", "xgb", "bert")   # every model in the four-model tables (design D1 of add-bert-qwen-ensemble)
ZERO_SHOT = ("jev", "qwen")                # no training labels
SUPERVISED = ("xgb", "bert")               # trained on SageMaker, one fitted model per split repeat
# Models with one prediction per item, independent of the split repeat. `qwen_think` (thinking
# enabled) is scored only on its subsample and is never part of a four-model table.
ITEM_MODELS = ("jev", "qwen", "qwen_think")
BACKENDS = ("jev",)       # the only model queried from this machine
MODES = ("mock", "live")


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load `configs/<name>.yaml`."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f) or {}


def results_dir(mode: str, *parts: str) -> Path:
    p = RESULTS.joinpath(mode, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_env() -> None:
    """Load `.env` from the project root; variables already exported win."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
