"""Mock mode: synthetic "mock-xgb", "mock-bert", "mock-qwen" and "mock-qwen_think" predictions,
in the same files the SageMaker runs produce. No model is trained or queried."""
from __future__ import annotations

import pandas as pd

from .. import backends
from ..data.datasets import Dataset
from ..results import model_dir, prob_columns, write_item_predictions, write_supervised_predictions

FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)


def label_budget(ds: Dataset, repeat: int = 0) -> dict:
    r = ds.repeats[repeat]
    return {"training": len(r["train"]), "early_stopping_and_selection": len(r["val"]), "threshold_selection": len(r["val"]) if ds.binary else 0,
            "calibration": 0}


def write_mock_supervised(ds: Dataset, model: str) -> None:
    mock = backends.mock_model(ds.name, model)
    cols = prob_columns(len(ds.classes))
    parts, curve = [], []
    for r, rep in enumerate(ds.repeats):
        for split in ("val", "test"):
            f = pd.DataFrame(mock.predict_ids(rep[split], r), columns=cols)
            f.insert(0, "split", split), f.insert(0, "repeat", r), f.insert(0, "id", rep[split])
            parts.append(f)
        for frac in FRACTIONS:
            f = pd.DataFrame(mock.predict_ids(rep["test"], r, frac), columns=cols)
            f.insert(0, "n_train", round(frac * len(rep["train"]))), f.insert(0, "fraction", frac)
            f.insert(0, "repeat", r), f.insert(0, "id", rep["test"])
            curve.append(f)
    run = {"model_version": mock.model_version, "label_budget": [label_budget(ds, r) for r in range(len(ds.repeats))],
           "fractions": list(FRACTIONS), "note": f"synthetic mock-{model} predictions; no model was trained",
           # synthetic cost and timing, so that the cost tables and sentences can be built in mock mode
           "instance_type": "ml.m5.xlarge", "region": "us-east-2", "cost_usd": 0.01, "billable_seconds": 150.0,
           "infer_s_per_item": 1e-4}
    write_supervised_predictions("mock", ds, model, pd.concat(parts, ignore_index=True), run)
    pd.concat(curve, ignore_index=True).to_parquet(model_dir("mock", ds.name, model) / "learning_curve.parquet", index=False)


def write_mock_zero_shot(ds: Dataset, model: str, ids: list[str]) -> None:
    mock = backends.mock_model(ds.name, model)
    f = pd.DataFrame(mock.predict_ids(ids, "item"), columns=prob_columns(len(ds.classes)))
    f.insert(0, "valid", True), f.insert(0, "id", ids)
    f["latency_s"] = mock.p["latency_s"]
    if model == "qwen_think":
        f["thinking_tokens"], f["thinking_truncated"], f["seconds"] = 500, False, mock.p["latency_s"]
    secs = len(ids) * mock.p["latency_s"]
    write_item_predictions("mock", ds, model, f, {
        "model_version": mock.model_version, "note": f"synthetic mock-{model} predictions; no model was queried",
        # synthetic cost and timing, so that the cost tables and sentences can be built in mock mode
        "region": "us-east-2", "instance_type": "ml.m5.xlarge", "load_s": 20.0,
        "jobs": {"Infer": {"instance_type": "ml.m5.xlarge", "cost_usd": 0.05, "billable_seconds": 600.0}},
        "timing": {"nothink": {"items": len(ids), "seconds": secs}, "latency_s": [mock.p["latency_s"]] * 3},
        "run_totals": {"nothink_items": len(ids), "nothink_seconds": secs, "think_seconds": 0.0, "latency_seconds": 0.0}})


def write_mock_baseline(ds: Dataset) -> None:
    from ..llm import thinking_subsample
    for model in ("xgb", "bert"):
        write_mock_supervised(ds, model)
    from ..escalation_val import ids as escalation_ids
    write_mock_zero_shot(ds, "qwen", list(ds.jev_ids) + escalation_ids(ds))
    write_mock_zero_shot(ds, "qwen_think", thinking_subsample(ds))
