"""`importance` stage: estimating Jev's accuracy from a small labelling budget (design D9).

The pool is a fixed set of examples with cached Jev predictions. The sampling procedure sees
only those predictions. It draws n items with replacement, uniformly or from the
confidence-guided proposal

    q_i = (1 - eps) * u_i / sum(u) + eps / N,    u_i = (1 - p_max,i)^alpha,

and asks a label oracle for the labels of the drawn items only. The error rate is estimated
with the Hansen-Hurwitz estimator, mean of loss_i / (N q_i) over the draws, which is unbiased
for any q > 0; accuracy is one minus it. The proposal follows the predicted uncertainty
because that is the model's own estimate of its loss (active risk estimation). Full-pool
accuracy is used only afterwards, as the ground truth the estimates are compared with.
"""
from __future__ import annotations

import numpy as np

from .. import config, results
from ..data import datasets
from ..data.datasets import Dataset
from .common import plt, save_json, savefig, selected

STAGE = "importance"
Z95 = 1.959964


class LabelOracle:
    """Holds the pool's labels and reveals them only for the indices it is asked about."""

    def __init__(self, labels: np.ndarray):
        self._labels = np.asarray(labels)
        self.calls = 0
        self.revealed: set[int] = set()

    def reveal(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        self.calls += int(idx.size)
        self.revealed.update(np.unique(idx).tolist())
        return self._labels[idx]


def proposal(P: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    """Sampling probabilities from the predictions alone; strictly positive and summing to one."""
    u = (1.0 - P.max(1)) ** alpha
    n = len(u)
    base = u / u.sum() if u.sum() > 0 else np.full(n, 1.0 / n)
    return (1 - eps) * base + eps / n


def estimate(P: np.ndarray, pred: np.ndarray, oracle: LabelOracle, q: np.ndarray, n: int, trials: int, rng) -> dict:
    """`trials` independent estimates of accuracy from n labelled draws each, with 95% interval bounds."""
    N = len(q)
    idx = rng.choice(N, size=(trials, n), p=q)
    labels = oracle.reveal(idx)                                  # the only labels the procedure ever sees
    loss = (pred[idx] != labels).astype(float)                   # an invalid prediction (-1) is an error
    z = loss / (N * q[idx])                                      # inverse-probability weights: uniform target over proposal
    err = z.mean(1)
    se = np.sqrt(z.var(1, ddof=1) / n)
    acc = 1.0 - err
    return {"accuracy": acc, "lo": acc - Z95 * se, "hi": acc + Z95 * se}


def summarize(e: dict, truth: float) -> dict:
    est = e["accuracy"]
    return {"mean": float(est.mean()), "bias": float(est.mean() - truth), "variance": float(est.var()),
            "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
            "coverage95": float(np.mean((e["lo"] <= truth) & (truth <= e["hi"])))}


def pool(ds: Dataset, mode: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Every example Jev was evaluated on: all rows of a repeated-split dataset, or the held-out split."""
    ids = ds.jev_ids if ds.protocol == "repeated" else ds.eval_ids
    df = results.read_predictions(mode, ds.name, "jev").set_index("id").loc[ids]
    P = df[results.prob_columns(len(ds.classes))].to_numpy(float)
    pred = np.where(df.valid.to_numpy(bool), df.label.to_numpy(int), -1)
    return ids, P, pred, df.y.to_numpy(int), sorted(df["mode"].unique().tolist())


def run_dataset(ds: Dataset, mode: str) -> dict:
    cfg = config.load("importance")
    ids, P, pred, y, modes = pool(ds, mode)
    N = len(ids)
    budgets = [n for n in cfg["budgets"] if n < N]
    truth = float(np.mean(pred == y))            # ground truth for the comparison only
    oracle = LabelOracle(y)
    draws = 0

    def table(q: np.ndarray, stream: int) -> dict:
        nonlocal draws
        out = {}
        for n in budgets:
            rng = np.random.default_rng([cfg["seed"], stream, n])
            out[str(n)] = summarize(estimate(P, pred, oracle, q, n, cfg["trials"], rng), truth)
            draws += n * cfg["trials"]
        return out

    q_is = proposal(P, cfg["alpha"], cfg["epsilon"])
    samplers = {"uniform": table(np.full(N, 1.0 / N), 0), "importance": table(q_is, 1)}
    sens = {str(a): table(proposal(P, a, cfg["epsilon"]), 2 + i) for i, a in enumerate(cfg["alpha_sensitivity"])}
    if oracle.calls != draws:
        raise RuntimeError("the label oracle was asked for labels that were not drawn")
    ratio = {n: (samplers["uniform"][n]["variance"] / samplers["importance"][n]["variance"]
                 if samplers["importance"][n]["variance"] > 0 else None) for n in samplers["uniform"]}
    # Diagnostics for reading the result (computed after the fact; not available to the sampler).
    u, wrong = 1.0 - P.max(1), pred != y
    zero = u <= 1e-9
    diag = {"errors": int(wrong.sum()), "items_with_zero_reported_uncertainty": int(zero.sum()),
            "errors_with_zero_reported_uncertainty": int((wrong & zero).sum()),
            "mean_predicted_uncertainty": float(u.mean()), "error_rate": float(wrong.mean()),
            "share_of_sampling_probability_on_errors": float(q_is[wrong].sum()),
            "distinct_probability_values": int(len(np.unique(np.round(P.max(1), 6))))}
    res = {"dataset": ds.name, "title": ds.title, "pool": {"n": N, "accuracy": truth, **diag},
           "estimated_quantity": "accuracy", "trials": cfg["trials"], "budgets": budgets, "epsilon": cfg["epsilon"],
           "alpha": cfg["alpha"], "q": {"min": float(q_is.min()), "max": float(q_is.max()), "sum": float(q_is.sum())},
           "samplers": samplers, "variance_ratio_uniform_over_importance": ratio,
           "alpha_sensitivity": {a: {n: t[n]["rmse"] for n in t} for a, t in sens.items()},
           "alpha_variance_ratio_uniform_over_importance": {
               a: {n: (samplers["uniform"][n]["variance"] / t[n]["variance"] if t[n]["variance"] > 0 else None) for n in t}
               for a, t in sens.items()},
           "label_oracle": {"labels_requested": oracle.calls, "draws": draws}}
    save_json(mode, ds.name, STAGE, "importance", res, modes)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for name, col, label in (("uniform", "C0", "uniform sampling"), ("importance", "C3", "confidence-guided sampling")):
        ax.plot(budgets, [samplers[name][str(n)]["rmse"] for n in budgets], "o-", c=col, ms=4, label=label)
    for a, t in sens.items():
        if float(a) != cfg["alpha"]:
            ax.plot(budgets, [t[str(n)]["rmse"] for n in budgets], ":", c="C3", alpha=0.6, lw=1, label=f"confidence-guided, alpha={a}")
    ax.set(xscale="log", xlabel="label budget n", ylabel="RMSE of the accuracy estimate", title=f"{ds.title} (pool N={N})")
    ax.set_xticks(budgets, [str(n) for n in budgets])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(fontsize=6)
    savefig(fig, mode, ds.name, STAGE, "rmse_vs_budget")
    return res


def run(args) -> int:
    for name in selected(args):
        res = run_dataset(datasets.load(name), args.mode)
        top = str(res["budgets"][-1])
        r = res["variance_ratio_uniform_over_importance"][top]
        print(f"importance ({args.mode}): {name}: pool {res['pool']['n']}, accuracy {res['pool']['accuracy']:.3f}; at n={top} "
              f"RMSE uniform {res['samplers']['uniform'][top]['rmse']:.4f}, confidence-guided "
              f"{res['samplers']['importance'][top]['rmse']:.4f}, variance ratio {r:.2f}; zero remote calls")
    return 0
