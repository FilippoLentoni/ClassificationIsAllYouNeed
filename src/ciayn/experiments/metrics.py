"""`metrics` stage: classification metrics and calibration for every dataset and model (design D7).

Every model (Jev, Qwen, XGBoost, BERT) is scored on the identical examples of each split
repeat; the stage stops if any two identifier sets differ. Qwen with thinking enabled is scored
separately, on its subsample only, next to the other models on the same identifiers. Held-out datasets get bootstrap 95% intervals;
repeated-split datasets get the mean and standard deviation over the repeats. Invalid or
missing outputs carry the INVALID label and a uniform distribution, so they are errors in
every metric.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import config, results
from ..data import datasets
from ..data.datasets import Dataset
from .common import MODEL_COLOUR, MODEL_LABEL, mean_sd, plt, save_json, savefig, selected

STAGE = "metrics"
HEADLINE = ("accuracy", "macro_f1", "roc_auc", "pr_auc", "ece", "brier", "log_loss")


def ece_bins() -> int:
    return int(config.load("metrics")["ece_bins"])


def accuracy(y, pred) -> float:
    return float(np.mean(pred == y))


def confusion(y, pred, k: int) -> np.ndarray:
    """k x k counts of (true, predicted) over valid predictions; invalid outputs are in no column."""
    ok = pred >= 0
    return np.bincount(y[ok] * k + pred[ok], minlength=k * k).reshape(k, k)


def per_class(y, pred, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm = confusion(y, pred, k)
    tp, support = np.diag(cm).astype(float), np.bincount(y, minlength=k).astype(float)
    predicted = cm.sum(0).astype(float)
    prec = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * tp, predicted + support, out=np.zeros(k), where=predicted + support > 0)
    return prec, rec, f1, support


def macro_f1(y, pred, k: int) -> float:
    return float(per_class(y, pred, k)[2].mean())


def _roc_auc(pos, score) -> float:
    """Mann-Whitney ROC-AUC with tied scores counted as one half."""
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _average_precision(pos, score) -> float:
    """Step-wise average precision over distinct thresholds (same definition as scikit-learn)."""
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], pos[order].astype(float)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, n = np.cumsum(t)[last], last + 1
    recall = tp / t.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * tp / n))


def auc_scores(y, P) -> tuple[float, float]:
    """Binary: ROC-AUC and PR-AUC of the positive (second) class. Multiclass: one-vs-rest macro over classes present."""
    n, k = P.shape
    counts = np.bincount(y, minlength=k)
    present = np.flatnonzero((counts > 0) & (counts < n))
    if len(present) == 0:
        return float("nan"), float("nan")
    if k == 2:
        return _roc_auc(y == 1, P[:, 1]), _average_precision(y == 1, P[:, 1])
    ranks = rankdata(P[:, present], axis=0)                      # all classes at once
    own = ranks[np.arange(n)[:, None], np.arange(len(present))[None, :]] * (y[:, None] == present[None, :])
    n1 = counts[present].astype(float)
    roc = (own.sum(0) - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return float(roc.mean()), float(np.mean([_average_precision(y == c, P[:, c]) for c in present]))


def ece(y, P, pred, bins: int | None = None) -> float:
    """Expected calibration error on max-probability confidence, equal-width bins."""
    bins = bins or ece_bins()
    conf, correct = P.max(1), (pred == y).astype(float)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1], right=True), 0, bins - 1)
    gap = np.abs(np.bincount(idx, weights=correct, minlength=bins) - np.bincount(idx, weights=conf, minlength=bins))
    return float(gap.sum() / len(y))


def brier(y, P) -> float:
    return float(np.mean(np.sum((P - np.eye(P.shape[1])[y]) ** 2, axis=1)))


LOG_LOSS_CLIP = 1e-12   # a reported probability of exactly zero for the true class is clipped here


def logloss(y, P) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], LOG_LOSS_CLIP, 1))))


def reliability(y, P, pred, bins: int | None = None) -> list[dict]:
    bins = bins or ece_bins()
    conf, correct = P.max(1), (pred == y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    return [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(np.sum(idx == b)),
             "accuracy": float(correct[idx == b].mean()), "confidence": float(conf[idx == b].mean())}
            for b in range(bins) if np.any(idx == b)]


def all_metrics(y, P, pred) -> dict:
    """Every headline metric from one set of predictions."""
    roc, pr = auc_scores(y, P)
    return {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, P.shape[1]), "roc_auc": roc, "pr_auc": pr,
            "ece": ece(y, P, pred), "brier": brier(y, P), "log_loss": logloss(y, P)}


def bootstrap_all(y, P, pred, n: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% percentile bootstrap intervals of every headline metric, from one pass over the resamples."""
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in HEADLINE}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for name, v in all_metrics(y[idx], P[idx], pred[idx]).items():
            if np.isfinite(v):
                draws[name].append(v)
    return {name: tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if v else (float("nan"), float("nan"))
            for name, v in draws.items()}


def evaluate(ds: Dataset, per_repeat: list[results.Scores]) -> dict:
    """Headline metrics with uncertainty, per-class metrics, confusion matrix and reliability for one model."""
    k, cfg = len(ds.classes), config.load("metrics")
    if ds.protocol == "heldout":
        s = per_repeat[0]
        point, ci = all_metrics(s.y, s.P, s.pred), bootstrap_all(s.y, s.P, s.pred, cfg["bootstrap"], cfg["seed"])
        headline = {name: {"value": float(point[name]), "ci95": ci[name]} for name in HEADLINE}
        uncertainty = f"95% bootstrap interval, {cfg['bootstrap']} resamples"
    else:
        rows = [all_metrics(s.y, s.P, s.pred) for s in per_repeat]
        headline = {name: mean_sd(r[name] for r in rows) for name in HEADLINE}
        uncertainty = f"mean and standard deviation over {len(per_repeat)} split repeats"
    y, P, pred = (np.concatenate([getattr(s, a) for s in per_repeat]) for a in ("y", "P", "pred"))
    prec, rec, f1, support = per_class(y, pred, k)
    classes = [{"class": c, "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                "support": int(support[i])} for i, c in enumerate(ds.classes)]
    invalid = pred < 0
    return {"uncertainty": uncertainty, "headline": headline, "n_evaluated": int(len(per_repeat[0])),
            "n_repeats": len(per_repeat), "invalid_rate": float(invalid.mean()),
            "mean_confidence": float(P.max(1).mean()),
            "predicted_share": (np.bincount(pred[~invalid], minlength=k) / len(pred)).tolist(),
            "true_share": (np.bincount(y, minlength=k) / len(y)).tolist(),
            "per_class": classes, "confusion": {"rows_true_cols_pred": confusion(y, pred, k).tolist(),
                                                "pooled_over_repeats": len(per_repeat) > 1},
            "reliability": reliability(y, P, pred)}


def label_budget(ds: Dataset, mode: str, models: list[str]) -> dict:
    """Labels each model used for training, calibration and threshold selection (per split repeat).

    Zero-shot models use no training labels; their threshold is selected on Jev's validation items.
    """
    zero = {"training": 0, "early_stopping_and_selection": 0, "calibration": 0,
            "threshold_selection": len(ds.jev_split("val")) if ds.binary else 0}
    out = {}
    for m in models:
        if m in config.ZERO_SHOT:
            out[m] = dict(zero)
            continue
        run = json.loads((results.model_dir(mode, ds.name, m) / "run.json").read_text())
        b = run["label_budget"][0] if "label_budget" in run else run["repeats"]["0"]["label_budget"]
        out[m] = {k: int(b[k]) for k in zero}
    return out


def learning_curve(ds: Dataset, mode: str, model: str = "xgb") -> list[dict]:
    path = results.model_dir(mode, ds.name, model) / "learning_curve.parquet"
    curve = pd.read_parquet(path)
    k, cols, rows = len(ds.classes), results.prob_columns(len(ds.classes)), []
    for frac, part in curve.groupby("fraction"):
        acc, f1, sizes = [], [], []
        for r, g in part.groupby("repeat"):
            g = g.set_index("id").loc[[i for i in ds.ids("test", int(r))]]
            y, pred = ds.labels(list(g.index)), g[cols].to_numpy().argmax(1)
            acc.append(accuracy(y, pred)), f1.append(macro_f1(y, pred, k)), sizes.append(int(g["n_train"].iloc[0]))
        rows.append({"fraction": float(frac), "n_train": int(round(np.mean(sizes))), "n_repeats": len(acc),
                     "accuracy": mean_sd(acc), "macro_f1": mean_sd(f1)})
    return rows


def thinking_subsample(ds: Dataset, mode: str, per_model: dict) -> dict | None:
    """Qwen with thinking enabled, and every other model, on the thinking subsample only (repeat 0)."""
    if not results.pred_path(mode, ds.name, "qwen_think").exists():
        return None
    think = results.read_predictions(mode, ds.name, "qwen_think")
    have = set(think["id"])
    ids = [i for i in ds.ids("test", 0) if i in have]      # evaluation items only (a run on every item also has validation items)
    if not ids:
        raise RuntimeError(f"{ds.name}: thinking predictions contain no evaluation example")
    k = len(ds.classes)
    s = results._to_scores(think, ids, k)
    out = {"n": len(ids), "models": {"qwen_think": all_metrics(s.y, s.P, s.pred) | {"mean_confidence": float(s.conf.mean())}}}
    for m, reps in per_model.items():
        t = results.subset(reps[0], ids)
        out["models"][m] = all_metrics(t.y, t.P, t.pred) | {"mean_confidence": float(t.conf.mean())}
    out["invalid_rate_qwen_think"] = float((~s.valid).mean())
    return out


def reliability_figure(ds: Dataset, res: dict, mode: str) -> None:
    fig, axes = plt.subplots(1, len(config.MODELS), figsize=(3.2 * len(config.MODELS), 2.9), sharey=True)
    bins = ece_bins()
    for ax, model in zip(axes, config.MODELS):
        if model not in res["models"]:
            ax.text(0.5, 0.5, f"{MODEL_LABEL[model]}\nnot available", ha="center", va="center", fontsize=8)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            continue
        rel = res["models"][model]["reliability"]
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
        ax.bar([(b["lo"] + b["hi"]) / 2 for b in rel], [b["accuracy"] for b in rel], width=1 / bins, alpha=0.7,
               color=MODEL_COLOUR[model], edgecolor="k", lw=0.4)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence (max class probability)",
               title=f"{MODEL_LABEL[model]}\nECE {res['models'][model]['headline']['ece']['value']:.3f}")
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("accuracy")
    fig.suptitle(ds.title, fontsize=10)
    savefig(fig, mode, ds.name, STAGE, "reliability")


def learning_curve_figure(ds: Dataset, res: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for model, rows in res["learning_curves"].items():
        n, acc = [r["n_train"] for r in rows], [r["accuracy"]["value"] for r in rows]
        ax.errorbar(n, acc, yerr=[r["accuracy"]["sd"] for r in rows], marker="o", ms=4, c=MODEL_COLOUR[model], capsize=2,
                    label=MODEL_LABEL[model])
    for model in config.ZERO_SHOT:
        if model in res["models"]:
            ax.axhline(res["models"][model]["headline"]["accuracy"]["value"], c=MODEL_COLOUR[model], ls="--",
                       label=f"{MODEL_LABEL[model]}, 0 training labels")
    ax.set(xscale="log", xlabel="training labels", ylabel="accuracy on evaluation examples", title=ds.title)
    ax.legend(fontsize=7)
    savefig(fig, mode, ds.name, STAGE, "learning_curve")


def run_dataset(ds: Dataset, mode: str) -> dict:
    """Jev is required. A model that has no predictions yet is reported as missing, never filled in."""
    models = [m for m in config.MODELS if m == "jev" or results.pred_path(mode, ds.name, m).exists()]
    missing = [m for m in config.MODELS if m not in models]
    frames = {m: results.read_predictions(mode, ds.name, m) for m in models}
    per_model = {m: [results.scores(mode, ds, m, "test", r, frames[m]) for r in range(len(ds.repeats))] for m in models}
    for m in models[1:]:                      # every model against Jev, hence every pair
        for r, (a, b) in enumerate(zip(per_model["jev"], per_model[m])):
            results.same_examples(a, b, f"{ds.name} repeat {r}, jev vs {m}")
    curves = {m: learning_curve(ds, mode, m) for m in models
              if m in config.SUPERVISED and (results.model_dir(mode, ds.name, m) / "learning_curve.parquet").exists()}
    res = {"dataset": ds.name, "title": ds.title, "protocol": ds.protocol, "classes": ds.classes, "positive": ds.positive,
           "ece_bins": ece_bins(), "models": {m: evaluate(ds, per_model[m]) for m in models}, "missing_models": missing,
           "model_version": {m: results.read_summary(mode, ds.name, m)["model_version"] for m in models},
           "label_budget": label_budget(ds, mode, models),
           "learning_curves": curves, "thinking_subsample": thinking_subsample(ds, mode, per_model)}
    modes = sorted({x for m in models for s in per_model[m] for x in s.modes})
    out = config.results_dir(mode, ds.name, STAGE)
    for m in config.MODELS:
        if m in models:
            pd.DataFrame(res["models"][m]["per_class"]).to_csv(out / f"per_class_{m}.csv", index=False)
        else:
            (out / f"per_class_{m}.csv").unlink(missing_ok=True)
    save_json(mode, ds.name, STAGE, "metrics", res, modes)
    reliability_figure(ds, res, mode)
    if res["learning_curves"]:
        learning_curve_figure(ds, res, mode)
    else:
        (out / "learning_curve.pdf").unlink(missing_ok=True)
    return res


def run(args) -> int:
    for name in selected(args):
        res = run_dataset(datasets.load(name), args.mode)
        line = "; ".join(f"{m}: acc {res['models'][m]['headline']['accuracy']['value']:.3f} "
                         f"macro-F1 {res['models'][m]['headline']['macro_f1']['value']:.3f} "
                         f"ECE {res['models'][m]['headline']['ece']['value']:.3f}" for m in res["models"])
        if res["missing_models"]:
            line += f"; NOT AVAILABLE: {', '.join(res['missing_models'])} (no predictions yet)"
        print(f"metrics ({args.mode}): {name}: n={res['models']['jev']['n_evaluated']} x {res['models']['jev']['n_repeats']} repeat(s); {line}")
    return 0
