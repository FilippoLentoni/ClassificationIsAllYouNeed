"""`thresholding` stage: threshold selection for binary tasks (design D8).

Decision rule: predict the positive class when P(positive | x) >= t. The positive class is
the second class of a binary dataset. Three thresholds are reported: the default 0.5, the
analytic cost threshold C_FP / (C_FP + C_FN), and the threshold that minimizes mean decision
cost on validation examples. Invalid or missing outputs are errors under every threshold:
they count as the wrong decision for their row.
"""
from __future__ import annotations

import numpy as np

from .. import config

STAGE = "thresholding"
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
DEFAULT_T = 0.5


def cost_matrix(dataset: str, classes: list[str] | None = None) -> np.ndarray:
    cfg = config.load("costs")[dataset]
    if classes is not None and list(cfg["classes"]) != list(classes):
        raise ValueError(f"costs.yaml class order {cfg['classes']} != dataset order {classes}")
    return np.asarray(cfg["matrix"], float)


def cost_is_illustrative(dataset: str) -> bool:
    return not config.load("costs")[dataset].get("canonical", False)


def analytic_threshold(C: np.ndarray) -> float:
    """C_FP / (C_FP + C_FN), with class 1 as the positive class."""
    c_fp, c_fn = C[0, 1], C[1, 0]
    return float(c_fp / (c_fp + c_fn))


def decide(p_pos: np.ndarray, t: float, y: np.ndarray | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    """Positive when P(positive | x) >= t; an invalid output becomes the wrong decision for its row."""
    d = (np.asarray(p_pos) >= t - 1e-12).astype(int)
    if valid is not None and y is not None:
        d = np.where(valid, d, 1 - y)
    return d


def at_threshold(y: np.ndarray, p_pos: np.ndarray, t: float, C: np.ndarray, valid: np.ndarray | None = None) -> dict:
    d = decide(p_pos, t, y, valid)
    tp, fp = np.sum((d == 1) & (y == 1)), np.sum((d == 1) & (y == 0))
    fn, tn = np.sum((d == 0) & (y == 1)), np.sum((d == 0) & (y == 0))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else 0.0
    return {"t": float(t), "precision": float(prec), "recall": float(rec), "f1_positive": float(f1),
            "accuracy": float((tp + tn) / len(y)), "macro_f1": float((f1 + f1_neg) / 2),
            "mean_cost": float(C[y, d].mean()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def sweep(y, p_pos, C, valid=None, grid: np.ndarray = GRID) -> dict[str, list[float]]:
    rows = [at_threshold(y, p_pos, t, C, valid) for t in grid]
    return {k: [r[k] for r in rows] for k in rows[0]}


def select_threshold(y_val, p_val, C, valid=None, grid: np.ndarray = GRID) -> float:
    """Validation labels only. Minimizes mean cost; ties go to the threshold closest to the analytic one, then smaller."""
    cost = np.array([at_threshold(y_val, p_val, t, C, valid)["mean_cost"] for t in grid])
    best = np.flatnonzero(np.isclose(cost, cost.min(), rtol=0, atol=1e-12))
    t_star = analytic_threshold(C)
    return float(min(grid[best], key=lambda t: (round(abs(t - t_star), 12), t)))


def three_thresholds(y_val, p_val, y_test, p_test, C, valid_val=None, valid_test=None) -> dict:
    ts = {"default": DEFAULT_T, "analytic": analytic_threshold(C),
          "validation": select_threshold(y_val, p_val, C, valid_val)}
    return {"thresholds": ts, "n_validation_labels": int(len(y_val)),
            "abs_diff_validation_vs_analytic": abs(ts["validation"] - ts["analytic"]),
            "at": {name: at_threshold(y_test, p_test, t, C, valid_test) for name, t in ts.items()}}


MARKERS = (("default", "default", "-.", "C1"), ("analytic", "analytic cost", "--", "C2"),
           ("validation", "validation-selected", ":", "C3"))


def figure(plt, y, p_pos, C, thresholds: dict[str, float], title: str, positive: str, valid=None):
    """Two panels: metric curves against t, and decision cost against t, with the three thresholds marked."""
    s = sweep(y, p_pos, C, valid)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for key, label, col in (("precision", "precision", "C0"), ("recall", "recall", "C4"),
                            ("accuracy", "accuracy", "C7"), ("f1_positive", f"F1 ({positive})", "C5")):
        a1.plot(GRID, s[key], c=col, lw=1.4, label=label)
    a1.plot(GRID, s["macro_f1"], c="k", ls=(0, (1, 1)), lw=1, label="macro-F1")
    a2.plot(GRID, s["mean_cost"], c="C0", lw=1.4, label="mean decision cost")
    for ax in (a1, a2):
        for name, label, ls, col in MARKERS:
            ax.axvline(thresholds[name], c=col, ls=ls, lw=1.6, label=f"{label} t = {thresholds[name]:.2f}")
        ax.set(xlabel=f"threshold t on P({positive})", xlim=(0, 1))
        ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    a1.set(ylabel="metric", ylim=(0, 1.02), title=title)
    a2.set(ylabel="mean decision cost", title="Decision cost")
    a1.title.set_fontsize(9), a2.title.set_fontsize(9)
    return fig


def run_dataset(ds, mode: str) -> dict:
    from .. import results
    from .common import MODEL_LABEL, mean_sd, plt, save_json, savefig

    C = cost_matrix(ds.name, ds.classes)
    res = {"dataset": ds.name, "title": ds.title, "positive": ds.positive, "rule": "predict positive when P(positive | x) >= t",
           "cost_matrix": {"classes": ds.classes, "rows_true_cols_pred": C.tolist(), "illustrative": cost_is_illustrative(ds.name)},
           "n_repeats": len(ds.repeats), "models": {}}
    modes = set()
    for model in [m for m in config.MODELS if m == "jev" or results.pred_path(mode, ds.name, m).exists()]:
        frame = results.read_predictions(mode, ds.name, model)
        reps = []
        for r in range(len(ds.repeats)):
            val, test = (results.scores(mode, ds, model, split, r, frame) for split in ("val", "test"))
            modes |= set(val.modes) | set(test.modes)
            reps.append(three_thresholds(val.y, val.P[:, 1], test.y, test.P[:, 1], C, val.valid, test.valid))
            if r == 0:   # the figure shows repeat 0; the tables give every repeat
                title = ds.title if len(ds.title) < 22 else "Breast Cancer Wisconsin"
                fig = figure(plt, test.y, test.P[:, 1], C, reps[0]["thresholds"],
                             f"{title}, {MODEL_LABEL[model]}", ds.positive, test.valid)
                savefig(fig, mode, ds.name, STAGE, f"threshold_sweep_{model}")
        summary = {"thresholds": {n: mean_sd(x["thresholds"][n] for x in reps) for n in ("default", "analytic", "validation")},
                   "abs_diff_validation_vs_analytic": mean_sd(x["abs_diff_validation_vs_analytic"] for x in reps),
                   "at": {n: {k: mean_sd(x["at"][n][k] for x in reps) for k in
                              ("precision", "recall", "f1_positive", "accuracy", "macro_f1", "mean_cost")}
                          for n in ("default", "analytic", "validation")},
                   # confusion counts summed over the split repeats (positive = the documented positive class)
                   "counts": {n: {k: int(sum(x["at"][n][k] for x in reps)) for k in ("tp", "fn", "fp", "tn")}
                              for n in ("default", "analytic", "validation")}}
        res["models"][model] = {"n_validation_labels": reps[0]["n_validation_labels"], "n_test": len(test), "summary": summary,
                                "repeats": reps}
    save_json(mode, ds.name, STAGE, "thresholding", res, sorted(modes))
    return res


def run(args) -> int:
    from ..data import datasets
    from .common import selected

    for name in selected(args):
        if name not in config.BINARY:
            continue
        res = run_dataset(datasets.load(name), args.mode)
        for model, m in res["models"].items():
            s = m["summary"]
            print(f"thresholding ({args.mode}): {name}/{model}: validation-selected t {s['thresholds']['validation']['value']:.2f} "
                  f"(analytic {s['thresholds']['analytic']['value']:.2f}); mean cost default "
                  f"{s['at']['default']['mean_cost']['value']:.3f}, analytic {s['at']['analytic']['mean_cost']['value']:.3f}, "
                  f"validation {s['at']['validation']['mean_cost']['value']:.3f}")
    return 0
