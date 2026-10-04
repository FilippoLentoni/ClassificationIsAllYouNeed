"""Package a dataset for the BERT training job: one directory per split repeat (design D2).

Every example is the text Jev receives (`serialize`), so BERT reads what Jev reads. `train.jsonl`
and `val.jsonl` carry labels; `test.jsonl` never does. `fractions.json` holds the row indices of
the nested training fractions, computed with the XGBoost job's own function and seed, so both
supervised models learn from the same examples at every size.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..baseline.scripts.train import nested_subset
from ..data.datasets import Dataset
from ..data.serialize import serialize


def job_spec(ds: Dataset) -> dict:
    cfg, base = config.load("encoder"), config.load("baseline")
    return {"dataset": ds.name, "n_classes": len(ds.classes), "n_repeats": len(ds.repeats),
            "model_id": cfg["model_id"], "revision": cfg["revision"], "grid": cfg["grid"], "max_epochs": cfg["max_epochs"],
            "batch_size": cfg["batch_size"], "eval_batch_size": cfg["eval_batch_size"], "warmup_ratio": cfg["warmup_ratio"],
            "weight_decay": cfg["weight_decay"], "seed": cfg["seed"], "max_length": cfg["max_length"][ds.name],
            "fractions": base["fractions"], "time_budget_s": cfg["curve_time_share"] * cfg["max_runtime_s"]}


def fraction_indices(ds: Dataset, repeat: int) -> dict[str, list[int]]:
    """Row indices into the repeat's training part for each training fraction (identical to XGBoost's)."""
    y = ds.y.loc[ds.repeats[repeat]["train"]].to_numpy()
    seed = int(config.load("baseline")["seed"]) + repeat
    return {str(f): nested_subset(y, f, seed).tolist() for f in config.load("baseline")["fractions"]}


def write_inputs(ds: Dataset, out: Path) -> Path:
    """Write <out>/<dataset>/{job.json, repeat_<r>/{train,val,test}.jsonl, repeat_<r>/fractions.json}."""
    root = out / ds.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "job.json").write_text(json.dumps(job_spec(ds), indent=1))
    for r, rep in enumerate(ds.repeats):
        d = root / f"repeat_{r}"
        d.mkdir(exist_ok=True)
        for split in ("train", "val", "test"):
            with open(d / f"{split}.jsonl", "w") as f:
                for i in rep[split]:
                    rec = {"id": i, "text": serialize(ds, i)}
                    if split != "test":                   # test examples are supplied without labels
                        rec["y"] = int(ds.y[i])
                    f.write(json.dumps(rec) + "\n")
        (d / "fractions.json").write_text(json.dumps(fraction_indices(ds, r)))
    return root
