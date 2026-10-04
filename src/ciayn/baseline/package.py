"""Package a dataset for the baseline pipeline: one directory per split repeat (design D6).

`train.csv` and `val.csv` carry labels; `test.csv` never does. For the held-out datasets this
means evaluation labels never leave this machine. For the repeated-split datasets a row that
is a test row in one repeat is a training row in another, so its label is in that other
repeat's directory, but each repeat is fitted only from its own directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config
from ..data.datasets import Dataset


def job_spec(ds: Dataset) -> dict:
    cfg = config.load("baseline")
    job = {"dataset": ds.name, "kind": ds.kind, "n_classes": len(ds.classes), "n_repeats": len(ds.repeats),
           "grid": cfg["grid"], "max_rounds": cfg["max_rounds"], "patience": cfg["patience"],
           "fractions": cfg["fractions"], "seed": cfg["seed"],
           "time_budget_s": cfg["curve_time_share"] * cfg["max_runtime_s"]["train"]}
    if ds.kind == "tabular":
        numeric = [c for c in ds.frame.columns if pd.api.types.is_numeric_dtype(ds.frame[c])]
        job.update(numeric=numeric, categorical=[c for c in ds.frame.columns if c not in numeric])
    else:
        job.update(ngram_range=cfg["text"]["ngram_range"], max_features=cfg["text"]["max_features"],
                   max_bin=cfg["text"]["max_bin"])
    return job


def write_inputs(ds: Dataset, out: Path) -> Path:
    """Write <out>/<dataset>/{job.json, repeat_<r>/{train,val,test}.csv}; returns the dataset directory."""
    root = out / ds.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "job.json").write_text(json.dumps(job_spec(ds), indent=1))
    for r, rep in enumerate(ds.repeats):
        d = root / f"repeat_{r}"
        d.mkdir(exist_ok=True)
        for split in ("train", "val", "test"):
            frame = ds.frame.loc[rep[split]].copy()
            frame.insert(0, "id", frame.index)
            if split != "test":                       # test examples are supplied without labels
                frame.insert(1, "y", ds.y.loc[rep[split]].to_numpy())
            frame.to_csv(d / f"{split}.csv", index=False)
    return root
