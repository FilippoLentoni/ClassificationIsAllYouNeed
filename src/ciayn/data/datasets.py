"""The five datasets: fixed ordered class sets, pinned sources and the split protocol (design D1).

Two protocols:

- ``repeated`` (German Credit, Breast Cancer Wisconsin Diagnostic): ``repeats`` seeded,
  stratified train/val/test splits over all rows. Jev predicts every row once, because each
  repeat's validation and test parts both need its predictions.
- ``heldout`` (SST-2, AG News): one "repeat" whose test part is the complete
  official held-out split; val is a seeded stratified draw from the official training split
  and train is the rest of it.

Splits are computed here from `configs/datasets.yaml`; `python -m ciayn freeze` writes them
to `results/splits/`, and `load()` refuses to return a dataset whose computed splits differ
from the frozen ones (see `manifest.py`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config

SPLITS = ("train", "val", "test")


@dataclass
class Dataset:
    name: str
    title: str
    kind: str                      # "tabular" | "text"
    protocol: str                  # "repeated" | "heldout"
    classes: list[str]             # fixed class order; probability vectors follow it
    positive: str | None           # positive class of a binary dataset (always classes[1])
    frame: pd.DataFrame            # indexed by example id; tabular: display-named features, text: "text"
    y: pd.Series                   # class index per example id
    repeats: list[dict[str, list[str]]]   # per repeat: train / val / test example ids
    jev_val: list[str]             # validation ids Jev is queried on (threshold selection)
    task: str = ""
    criteria: dict[str, str] = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.frame.columns) if self.kind == "tabular" else []

    @property
    def binary(self) -> bool:
        return len(self.classes) == 2

    def ids(self, split: str, repeat: int = 0) -> list[str]:
        return self.repeats[repeat][split]

    def labels(self, ids: list[str]) -> np.ndarray:
        return self.y.loc[ids].to_numpy()

    @property
    def eval_ids(self) -> list[str]:
        """Every example that is a test example in some repeat, in frame order."""
        if self.protocol == "heldout":
            return self.repeats[0]["test"]
        members = set().union(*(r["test"] for r in self.repeats))
        return [i for i in self.frame.index if i in members]

    @property
    def jev_ids(self) -> list[str]:
        """Examples Jev is queried on: every row (repeated splits), or the held-out split plus its validation items."""
        if self.protocol == "repeated":
            return list(self.frame.index)
        seen = set(self.eval_ids)
        return self.eval_ids + [i for i in self.jev_val if i not in seen]

    def jev_split(self, split: str, repeat: int = 0) -> list[str]:
        """The ids of a split that have Jev predictions (val is a subset for held-out datasets)."""
        if split == "val" and self.protocol == "heldout":
            return self.jev_val
        return self.ids(split, repeat)


def normalise_text(text: str) -> str:
    return " ".join(str(text).split())


def _stratified(ids: np.ndarray, y: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split off `size` ids stratified by y; returns (taken, rest)."""
    rest, taken = train_test_split(ids, test_size=size, stratify=y, random_state=seed)
    return taken, rest


def repeated_splits(ids: np.ndarray, y: pd.Series, n_repeats: int, fractions: dict) -> list[dict[str, list[str]]]:
    n_test, n_val = round(fractions["test"] * len(ids)), round(fractions["val"] * len(ids))
    out = []
    for r in range(n_repeats):
        test, rest = _stratified(ids, y.loc[ids].to_numpy(), n_test, r)
        val, train = _stratified(rest, y.loc[rest].to_numpy(), n_val, r)
        out.append({"train": list(train), "val": list(val), "test": list(test)})
    return out


def heldout_split(train_ids: np.ndarray, eval_ids: np.ndarray, y: pd.Series, val_size: int, seed: int):
    val, train = _stratified(train_ids, y.loc[train_ids].to_numpy(), val_size, seed)
    return [{"train": list(train), "val": list(val), "test": list(eval_ids)}]


# ---- raw sources -------------------------------------------------------------------------------

def _openml_frame(did: int) -> tuple[pd.DataFrame, pd.Series]:
    import openml

    (config.DATA / "openml").mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(config.DATA / "openml"))
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                    download_features_meta_data=False)
    X, y, _, _ = d.get_data(target=d.default_target_attribute)
    return X, y.astype(str)


def _german(name: str, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    X, y_raw = _openml_frame(cfg["source"]["openml"])
    X = X.copy()
    for col, mapping in (cfg.get("value_labels") or {}).items():
        X[col] = X[col].astype(str).map(lambda v, m=mapping: m.get(v, v))
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object:
            X[col] = X[col].astype("string")
    X = X.rename(columns=cfg["feature_names"])
    X.index = pd.Index([f"{name}-{i}" for i in range(len(X))], name="id")
    classes = list(cfg["classes"])
    y = pd.Series(y_raw.map(classes.index).to_numpy(), index=X.index, name="y")
    if y.isna().any():
        raise ValueError(f"{name}: labels outside the configured class set")
    return X, y.astype(int)


def _wdbc(name: str, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import load_breast_cancer

    b = load_breast_cancer()
    idx = pd.Index([f"{name}-{i}" for i in range(len(b.data))], name="id")
    X = pd.DataFrame(b.data, columns=[str(c) for c in b.feature_names], index=idx)
    names = [str(n) for n in b.target_names]                       # scikit-learn: 0 = malignant, 1 = benign
    y = pd.Series([cfg["classes"].index(names[t]) for t in b.target], index=idx, name="y")
    return X, y


def _text_table(name: str, cfg: dict) -> pd.DataFrame:
    """id, text, y, part ('train' | 'eval'), label_name; downloaded once and kept in cache/datasets/."""
    path = config.DATA / "datasets" / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    src, rows = cfg["source"], []
    if "hf" not in src:
        raise ValueError(f"{name}: text datasets are loaded from a pinned Hugging Face revision")
    from datasets import load_dataset

    hf = load_dataset(src["hf"], src.get("config"), revision=src["revision"], cache_dir=str(config.DATA / "hf"))
    for part, split in (("train", "train"), ("eval", cfg["eval_split"])):
        names = hf[split].features["label"].names
        if list(names) != list(cfg["classes"]):
            raise ValueError(f"{name}: source classes {names} != configured {cfg['classes']}")
        df = hf[split].to_pandas()
        rows += [(f"{name}-{split}-{i}", normalise_text(t), int(lab), part, names[int(lab)])
                 for i, (t, lab) in enumerate(zip(df[src["text_column"]], df["label"]))]
    table = pd.DataFrame(rows, columns=["id", "text", "y", "part", "label_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


@lru_cache(maxsize=None)
def build(name: str) -> Dataset:
    """The dataset with splits computed from configuration (not checked against the frozen manifest)."""
    cfg_all = config.load("datasets")
    cfg = cfg_all[name]
    classes, criteria, jev_val = list(cfg.get("classes") or []), dict(cfg.get("criteria") or {}), []
    if cfg["kind"] == "tabular":
        X, y = (_german if name == "german" else _wdbc)(name, cfg)
        repeats = repeated_splits(X.index.to_numpy(), y, cfg_all["repeats"], cfg_all["repeat_fractions"])
    else:
        table = _text_table(name, cfg).set_index("id")
        X, y = table[["text"]], table["y"].astype(int)
        repeats = heldout_split(table.index[table.part == "train"].to_numpy(),
                                table.index[table.part == "eval"].to_numpy(), y, cfg["val_size"], cfg_all["seed"])
        jev_val = repeats[0]["val"][: cfg.get("jev_val_size", 0)]
    y.name = "y"
    positive = cfg.get("positive")
    if positive is not None and classes.index(positive) != 1:
        raise ValueError(f"{name}: the positive class must be the second class")
    return Dataset(name, cfg["title"], cfg["kind"], cfg["protocol"], classes, positive, X, y, repeats, jev_val,
                   cfg["task"], criteria, dict(cfg["source"]))


@lru_cache(maxsize=None)
def load(name: str) -> Dataset:
    """The dataset, after checking that its splits are exactly the frozen ones."""
    from . import manifest

    ds = build(name)
    manifest.verify(ds)
    return ds


def clear_cache() -> None:
    build.cache_clear()
    load.cache_clear()
