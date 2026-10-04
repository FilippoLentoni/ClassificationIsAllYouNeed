"""Text serialization of examples for Jev.

Tabular rows become one `name: value` line per feature with decoded labels; missing values
are written as "unknown". Text examples are passed as their normalised text. Masked units
(explainability) are removed entirely: a feature line for tabular rows, a word for text, and
the tabular instructions state that fields not listed are unknown.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .datasets import Dataset

MISSING_NOTE = "Fields that are not listed are unknown."
MISSING_VALUE = "unknown"


def _fmt(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return MISSING_VALUE
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def units(ds: Dataset, ex_id: str) -> list[str]:
    """The maskable units of an example: feature names (tabular) or words (text)."""
    if ds.kind == "tabular":
        return ds.features
    return ds.frame.at[ex_id, "text"].split()


def serialize(ds: Dataset, ex_id: str, drop: Iterable[int] = ()) -> str:
    """Serialize an example, removing the units whose positions are in `drop`."""
    drop = set(drop)
    if ds.kind == "tabular":
        row = ds.frame.loc[ex_id]
        return "\n".join(f"{name}: {_fmt(row[name])}" for j, name in enumerate(ds.features) if j not in drop)
    return " ".join(w for j, w in enumerate(units(ds, ex_id)) if j not in drop)
