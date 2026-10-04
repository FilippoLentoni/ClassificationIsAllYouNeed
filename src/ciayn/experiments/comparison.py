"""`comparison` stage: F1 by training labels, cost per prediction and latency for the four models
(model-cost-latency-comparison; design D7 of add-bert-qwen-ensemble).

Every figure is computed from recorded quantities only: Jev's per-request cost (token counts at
its published price) and measured latency; SageMaker billable seconds at the configured list price
per instance hour; timings measured inside the jobs. Batch throughput is never reported as
per-request latency. A quantity that was not recorded is reported as None, never estimated.
"""
from __future__ import annotations

import json

import numpy as np

from .. import config, results
from ..data import datasets
from .common import MODEL_COLOUR, MODEL_LABEL, plt, read_json, save_json, savefig, selected

STAGE = "comparison"


def price(itype: str | None, region: str | None) -> float | None:
    if not itype or not region:
        return None
    cfg = config.load("baseline")
    return cfg["price_usd_per_hour"][itype] * cfg["region_price_factor"][region]


def _run(mode: str, ds: str, model: str) -> dict | None:
    p = results.model_dir(mode, ds, model) / "run.json"
    return json.loads(p.read_text()) if p.exists() else None


def jev_costs(mode: str, ds: str) -> dict:
    df = results.read_predictions(mode, ds, "jev")
    cost, lat = df["cost_usd"].to_numpy(float), df["latency_s"].to_numpy(float)
    known = np.isfinite(cost)
    return {"kind": "API, per request", "n_predictions": int(len(df)), "n_with_cost": int(known.sum()),
            "usd_per_1k": float(np.mean(cost[known]) * 1000) if known.any() else None,
            "how": "mean recorded cost per request (input tokens x published price), x 1,000",
            "latency": {"kind": "measured request latency", "mean_s": float(np.nanmean(lat)) if np.isfinite(lat).any() else None,
                        "median_s": float(np.nanmedian(lat)) if np.isfinite(lat).any() else None},
            "training_usd": 0.0}


def qwen_costs(mode: str, ds: str) -> dict:
    run = _run(mode, ds, "qwen") or {}
    t = run.get("timing") or {}
    nt = t.get("nothink") or {}
    infer = (run.get("jobs") or {}).get("Infer") or {}
    out = {"kind": "self-hosted, batch job", "training_usd": 0.0, "usd_per_1k": None, "usd_per_1k_inference_only": None,
           "latency": {"kind": "batch throughput and single-request latency", "throughput_items_per_s": None,
                       "single_request_mean_s": None, "single_request_median_s": None, "single_request_n": 0}}
    pr = price(infer.get("instance_type"), run.get("region"))
    if nt.get("items") and pr is not None:
        out["usd_per_1k_inference_only"] = pr / 3600 * nt["seconds"] / nt["items"] * 1000
        tot = run["run_totals"]
        other = pr / 3600 * (tot["think_seconds"] + tot["latency_seconds"])
        share = nt["seconds"] / tot["nothink_seconds"]            # this dataset's share of the job's scoring time
        out["usd_per_1k"] = (infer["cost_usd"] - other) * share / nt["items"] * 1000
        out["how"] = ("whole billed scoring job (instance start, weight copy, model loading included), less the time spent on "
                      "the thinking subsample and the latency sample, allocated to the dataset by its share of scoring time; "
                      "and inference time only, at the instance's list price")
        out["latency"]["throughput_items_per_s"] = nt["items"] / nt["seconds"]
        out["load_s"] = run.get("load_s")
        out["instance_type"], out["region"] = infer.get("instance_type"), run.get("region")
    lat = t.get("latency_s") or []
    if lat:
        out["latency"].update(single_request_mean_s=float(np.mean(lat)), single_request_median_s=float(np.median(lat)),
                              single_request_n=len(lat))
    return out


def supervised_costs(mode: str, ds: str, model: str) -> dict:
    run = _run(mode, ds, model) or {}
    per_item = run.get("infer_s_per_item") or ((run.get("repeats") or {}).get("0") or {}).get("infer_s_per_item")
    pr = price(run.get("instance_type"), run.get("region"))
    return {"kind": "trained on SageMaker", "training_usd": run.get("cost_usd"), "billable_seconds": run.get("billable_seconds"),
            "instance_type": run.get("instance_type"),
            "usd_per_1k": pr / 3600 * per_item * 1000 if per_item and pr is not None else None,
            "how": "measured batch inference time per evaluation item at the instance's list price, x 1,000; training cost is "
                   "the whole recorded execution",
            "latency": {"kind": "measured batch inference time per item", "per_item_s": per_item}}


def f1_by_labels(m: dict) -> dict:
    out = {}
    for model in config.MODELS:
        if model in config.SUPERVISED and model in m["learning_curves"]:
            out[model] = [{"n_train": c["n_train"], "macro_f1": c["macro_f1"]["value"], "sd": c["macro_f1"]["sd"]}
                          for c in m["learning_curves"][model]]
        elif model in config.ZERO_SHOT and model in m["models"]:
            out[model] = [{"n_train": 0, "macro_f1": m["models"][model]["headline"]["macro_f1"]["value"], "sd": 0.0}]
    return out


def figure(ds, curves: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for model, rows in curves.items():
        if model in config.SUPERVISED:
            ax.errorbar([r["n_train"] for r in rows], [r["macro_f1"] for r in rows], yerr=[r["sd"] for r in rows],
                        marker="o", ms=4, c=MODEL_COLOUR[model], capsize=2, label=MODEL_LABEL[model])
        else:
            ax.axhline(rows[0]["macro_f1"], c=MODEL_COLOUR[model], ls="--", label=f"{MODEL_LABEL[model]}, 0 labels")
    ax.set(xscale="log", xlabel="training labels", ylabel="macro-F1 on evaluation examples", title=ds.title)
    ax.legend(fontsize=6.5)
    savefig(fig, mode, ds.name, STAGE, "f1_by_labels")


def run_dataset(ds, mode: str) -> dict:
    m = read_json(mode, ds.name, "metrics", "metrics")
    curves = f1_by_labels(m)
    costs = {"jev": jev_costs(mode, ds.name)}
    if "qwen" in m["models"]:
        costs["qwen"] = qwen_costs(mode, ds.name)
    for model in config.SUPERVISED:
        if model in m["models"]:
            costs[model] = supervised_costs(mode, ds.name, model)
    res = {"dataset": ds.name, "title": ds.title, "f1_by_labels": curves, "costs": costs,
           "missing_models": m["missing_models"], "list_prices_usd_per_hour": config.load("baseline")["price_usd_per_hour"]}
    save_json(mode, ds.name, STAGE, "comparison", res, m["artifact_modes"])
    figure(ds, curves, mode)
    return res


def run(args) -> int:
    for name in selected(args):
        res = run_dataset(datasets.load(name), args.mode)
        parts = [f"{mo}: macro-F1 {rows[-1]['macro_f1']:.3f} at {rows[-1]['n_train']} labels" for mo, rows in res["f1_by_labels"].items()]
        print(f"comparison ({args.mode}): {name}: " + "; ".join(parts))
    return 0
