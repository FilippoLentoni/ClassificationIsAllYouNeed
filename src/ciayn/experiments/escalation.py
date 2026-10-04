"""`escalation` stage: Jev's low-confidence predictions escalated to Qwen3.6-27B (ensemble-escalation;
design D8 of add-bert-qwen-ensemble).

An item whose Jev maximum class probability is below a threshold t takes Qwen's prediction
(thinking disabled); every other item keeps Jev's. The curve runs over every distinct Jev
confidence, from escalating nothing (Jev alone) to escalating everything (Qwen alone). The
reported operating point uses the threshold that maximises macro-F1 on validation items only,
with ties broken towards fewer escalations; evaluation labels never influence it. Everything is
computed from saved predictions: no Jev request, no SageMaker job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, results
from ..data import datasets
from .common import MODEL_COLOUR, mean_sd, plt, read_json, save_json, savefig, selected
from .metrics import accuracy, macro_f1

STAGE = "escalation"
ALL = 1.01          # above every probability: escalate everything


def thresholds(conf: np.ndarray) -> np.ndarray:
    return np.unique(np.r_[0.0, np.unique(conf), ALL])


def ensemble(jev: results.Scores, qwen: results.Scores, t: float) -> tuple[np.ndarray, np.ndarray]:
    esc = jev.conf < t
    return np.where(esc, qwen.pred, jev.pred), esc


def curve(jev: results.Scores, qwen: results.Scores, ts: np.ndarray, k: int) -> list[dict]:
    out = []
    for t in ts:
        pred, esc = ensemble(jev, qwen, t)
        out.append({"t": float(t), "share_escalated": float(esc.mean()), "accuracy": accuracy(jev.y, pred),
                    "macro_f1": macro_f1(jev.y, pred, k)})
    return out


def select(val_curve: list[dict]) -> dict:
    """Highest validation macro-F1; ties go to the smaller share escalated (then the smaller t)."""
    return min(val_curve, key=lambda r: (-round(r["macro_f1"], 12), r["share_escalated"], r["t"]))


def unreachable(jev: results.Scores) -> dict:
    top = jev.conf >= jev.conf.max() - 1e-12
    errors = jev.pred != jev.y
    return {"max_confidence": float(jev.conf.max()), "items_at_max": int(top.sum()), "share_at_max": float(top.mean()),
            "errors": int(errors.sum()), "errors_at_max": int((errors & top).sum())}


def costs_at(share: float, comp: dict | None) -> dict:
    if not comp or "qwen" not in comp["costs"]:
        return {}
    j, q = comp["costs"]["jev"], comp["costs"]["qwen"]
    out = {}
    if j.get("usd_per_1k") is not None and q.get("usd_per_1k") is not None:
        out["usd_per_1k"] = j["usd_per_1k"] + share * q["usd_per_1k"]
        out["usd_per_1k_inference_only"] = j["usd_per_1k"] + share * q["usd_per_1k_inference_only"]
    jl, ql = j["latency"].get("mean_s"), q["latency"].get("single_request_mean_s")
    if jl is not None and ql is not None:
        out["mean_latency_s"] = jl + share * ql      # an escalated item waits for Jev, then for Qwen
    return out


def run_dataset(ds, mode: str) -> dict | None:
    if not results.pred_path(mode, ds.name, "qwen").exists():
        return None
    k = len(ds.classes)
    fj, fq = results.read_predictions(mode, ds.name, "jev"), results.read_predictions(mode, ds.name, "qwen")
    try:
        comp = read_json(mode, ds.name, "comparison", "comparison")
    except RuntimeError:
        comp = None
    reps, modes = [], set()
    from ..escalation_val import ids as escalation_ids
    esc = escalation_ids(ds)
    for r in range(len(ds.repeats)):
        if esc:            # AG News: the seeded validation items predicted for this purpose
            path = results.model_dir(mode, ds.name, "jev") / "escalation_val.parquet"
            if not path.exists():
                raise RuntimeError(f"missing {path}; run `python -m ciayn escalation-val --mode {mode}` first")
            vj, vq = results._to_scores(pd.read_parquet(path), esc, k), results._to_scores(fq, esc, k)
        else:
            vj, vq = results.scores(mode, ds, "jev", "val", r, fj), results.scores(mode, ds, "qwen", "val", r, fq)
        tj, tq = results.scores(mode, ds, "jev", "test", r, fj), results.scores(mode, ds, "qwen", "test", r, fq)
        results.same_examples(vj, vq, f"{ds.name} repeat {r} validation")
        results.same_examples(tj, tq, f"{ds.name} repeat {r} evaluation")
        modes |= set(vj.modes) | set(vq.modes) | set(tj.modes) | set(tq.modes)
        ts = thresholds(np.r_[vj.conf, tj.conf] if len(vj) else tj.conf)
        test_curve = curve(tj, tq, ts, k)
        rep = {"n_validation_labels": len(vj), "jev_alone": test_curve[0], "qwen_alone": test_curve[-1], "curve": test_curve,
               "unreachable": unreachable(tj)}
        if len(vj):
            chosen = select(curve(vj, vq, ts, k))
            rep.update(threshold=chosen["t"], validation_macro_f1=chosen["macro_f1"],
                       at=next(c for c in test_curve if c["t"] == chosen["t"]))
        else:                                   # never fall back to a threshold chosen any other way
            rep.update(threshold=None, at=None, no_threshold="Jev has no predictions on validation items for this dataset, "
                       "so no threshold can be selected without evaluation labels; only the curve is reported")
        reps.append(rep)
    selected_ok = all(r["at"] is not None for r in reps)
    summary = {key: mean_sd(r["at"][key] for r in reps) for key in ("accuracy", "macro_f1", "share_escalated")} if selected_ok else {}
    if selected_ok:
        summary["threshold"] = mean_sd(r["threshold"] for r in reps)
    for name in ("jev_alone", "qwen_alone"):
        summary[name] = {key: mean_sd(r[name][key] for r in reps) for key in ("accuracy", "macro_f1")}
    res = {"dataset": ds.name, "title": ds.title, "rule": "take Qwen's prediction when Jev's maximum class probability < t",
           "n_repeats": len(reps), "summary": summary, "repeats": reps,
           "threshold_selected": selected_ok, "no_threshold": None if selected_ok else reps[0]["no_threshold"],
           "costs_at_selected": costs_at(summary["share_escalated"]["value"], comp) if selected_ok else {},
           "costs_by_share": [{"share_escalated": c["share_escalated"], **costs_at(c["share_escalated"], comp)}
                              for c in reps[0]["curve"]],
           "unreachable": reps[0]["unreachable"]}
    save_json(mode, ds.name, STAGE, "escalation", res, sorted(modes))
    figure(ds, reps[0], mode)
    return res


def figure(ds, rep: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    c = rep["curve"]
    ax.plot([x["share_escalated"] for x in c], [x["macro_f1"] for x in c], c="k", lw=1.2, marker=".", ms=3)
    ax.scatter([c[0]["share_escalated"]], [c[0]["macro_f1"]], c=MODEL_COLOUR["jev"], zorder=3, label="Jev alone")
    ax.scatter([c[-1]["share_escalated"]], [c[-1]["macro_f1"]], c=MODEL_COLOUR["qwen"], zorder=3, label="Qwen alone")
    if rep["at"] is not None:
        ax.scatter([rep["at"]["share_escalated"]], [rep["at"]["macro_f1"]], marker="*", s=90, c="C3", zorder=4,
                   label=f"validation-selected t = {rep['threshold']:.2f}")
    ax.set(xlabel="share of items escalated to Qwen", ylabel="macro-F1 on evaluation examples", title=ds.title, xlim=(-0.02, 1.02))
    ax.legend(fontsize=6.5)
    savefig(fig, mode, ds.name, STAGE, "escalation_curve")


def run(args) -> int:
    for name in selected(args):
        res = run_dataset(datasets.load(name), args.mode)
        if res is None:
            print(f"escalation ({args.mode}): {name}: no Qwen predictions yet; skipped")
            continue
        s = res["summary"]
        point = (f"ensemble {s['macro_f1']['value']:.3f} escalating {s['share_escalated']['value']:.0%} "
                 f"(t = {s['threshold']['value']:.2f})" if res["threshold_selected"] else "no threshold selected (no validation items)")
        print(f"escalation ({args.mode}): {name}: Jev {s['jev_alone']['macro_f1']['value']:.3f}, Qwen "
              f"{s['qwen_alone']['macro_f1']['value']:.3f}, {point}; unreachable Jev errors "
              f"{res['unreachable']['errors_at_max']} of {res['unreachable']['errors']}")
    return 0
