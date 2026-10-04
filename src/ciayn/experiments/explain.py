"""`explain` stage: KernelSHAP and LIME on Jev's class probabilities, with a deletion test (design D10).

For each of a seeded sample of evaluation examples, a coalition z in {0, 1}^F keeps the units
with z_j = 1; a removed unit (a feature line, or a word) is dropped from the input. The
explained quantity is Jev's probability of the class it predicts on the full input. In live
mode every perturbed input is a Jev request: the worst-case count is checked against the
explain budget and the remaining cap before anything is sent, and every request goes through
the response cache, the ledger and the request log like any other.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import kendalltau

from .. import backends, config, results
from ..backends.budget import Budget, Ledger
from ..backends.cache import CachedBackend
from ..data import datasets
from ..data.datasets import Dataset
from ..data.serialize import serialize, units
from .common import plt, save_json, savefig, selected

log = logging.getLogger(__name__)
STAGE = "explain"
METHODS = ("kernel_shap", "lime")


def requests_per_example(cfg: dict) -> int:
    """Upper bound on model evaluations for one example: both methods plus the deletion test."""
    ks = len(cfg["deletion_k"])
    return cfg["shap_nsamples"] + 2 + cfg["lime_samples"] + len(METHODS) * ks + cfg["random_draws"] * ks


def choose_examples(ds: Dataset, cfg: dict) -> tuple[list[str], dict]:
    eligible = ds.eval_ids
    if ds.kind == "text":
        eligible = [i for i in ds.eval_ids if cfg["min_words"] <= len(units(ds, i)) <= cfg["max_words"]]
    rng = np.random.default_rng(cfg["seed"])
    pick = sorted(rng.choice(len(eligible), size=min(cfg["n_examples"], len(eligible)), replace=False))
    rule = {"eligible": len(eligible), "evaluation_examples": len(ds.eval_ids),
            "rule": (f"{cfg['min_words']} to {cfg['max_words']} words" if ds.kind == "text" else "all rows")}
    return [eligible[i] for i in pick], rule


class ProbabilityFunction:
    """Jev's P(predicted class | coalition) for one example, counting model evaluations."""

    def __init__(self, cb: CachedBackend, ds: Dataset, ex_id: str):
        self.cb, self.ds, self.ex = cb, ds, ex_id
        self.F = len(units(ds, ex_id))
        self.calls = 0
        if hasattr(cb.backend, "set_parent"):          # the mock needs to know whose masked inputs these are
            cb.backend.set_parent(ex_id)
        full = self.probs(np.ones((1, self.F)))[0]
        self.calls = 0
        self.label = int(np.argmax(full))
        self.p_full = float(full[self.label])

    def probs(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        self.calls += len(Z)
        texts = [serialize(self.ds, self.ex, drop=np.flatnonzero(z < 0.5)) for z in Z]
        return np.array([p.probs for p in self.cb.predict(self.ds, texts)])

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.probs(Z)[:, self.label]


def kernel_shap(f: ProbabilityFunction, nsamples: int, seed: int = 0) -> dict:
    import shap

    logging.getLogger("shap").setLevel(logging.WARNING)
    np.random.seed(seed)  # KernelExplainer samples coalitions from the global NumPy state
    f.calls = 0
    explainer = shap.KernelExplainer(f, np.zeros((1, f.F)))   # background: every unit removed
    phi = np.asarray(explainer.shap_values(np.ones((1, f.F)), nsamples=nsamples, l1_reg=False, silent=True)).reshape(-1)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return {"attributions": phi, "base_value": base, "additivity_error": abs(base + phi.sum() - f.p_full), "calls": f.calls}


def lime_explain(f: ProbabilityFunction, n: int, width: float, seed: int) -> dict:
    from lime.lime_base import LimeBase

    f.calls = 0
    rng = np.random.default_rng(seed)
    Z = np.ones((n, f.F))
    for row in Z[1:]:  # like LIME's text sampler: remove a uniformly drawn number of units
        row[rng.choice(f.F, rng.integers(1, f.F + 1), replace=False)] = 0
    dist = (1 - Z.sum(1) / np.sqrt(np.maximum(Z.sum(1), 1e-12) * f.F)) * 100   # cosine distance to all-ones
    base = LimeBase(lambda d: np.sqrt(np.exp(-(d ** 2) / width ** 2)), random_state=seed)
    _, exp, score, _ = base.explain_instance_with_data(Z, f(Z)[:, None], dist, 0, f.F, feature_selection="none")
    w = np.zeros(f.F)
    for j, v in exp:
        w[j] = v
    return {"attributions": w, "r2": float(score), "calls": f.calls}


def deletion(f: ProbabilityFunction, order: np.ndarray, k: int) -> tuple[float, bool]:
    z = np.ones(f.F)
    z[order[:k]] = 0
    p = f.probs(z)[0]
    return f.p_full - float(p[f.label]), int(np.argmax(p)) != f.label


def explain_example(f: ProbabilityFunction, cfg: dict, i: int) -> dict:
    ds = f.ds
    methods = {"kernel_shap": kernel_shap(f, cfg["shap_nsamples"], cfg["seed"] + i),
               "lime": lime_explain(f, cfg["lime_samples"], cfg["lime_kernel_width"], cfg["seed"] + i)}
    draw = np.random.default_rng([cfg["seed"], i])
    rand_orders = [draw.permutation(f.F) for _ in range(cfg["random_draws"])]
    faith = {}
    for k in cfg["deletion_k"]:
        rd = [deletion(f, o, k) for o in rand_orders]
        faith[str(k)] = {"rand_drop": float(np.mean([x for x, _ in rd])), "rand_flip": float(np.mean([y for _, y in rd]))}
        for m, res in methods.items():
            d, fl = deletion(f, np.argsort(-res["attributions"], kind="stable"), k)
            faith[str(k)][m] = {"top_drop": d, "top_flip": bool(fl)}
    a, b = methods["kernel_shap"]["attributions"], methods["lime"]["attributions"]
    top = lambda v: set(np.argsort(-v, kind="stable")[:3].tolist())  # noqa: E731
    tau = kendalltau(a, b).statistic
    names = units(ds, f.ex)
    return {"id": f.ex, "predicted": ds.classes[f.label], "gold": ds.classes[int(ds.y[f.ex])], "p_full": f.p_full,
            "n_units": f.F, "units": names, "kendall_tau": float(tau) if np.isfinite(tau) else None,
            "top3_overlap": len(top(a) & top(b)) / 3, "faithfulness": faith,
            **{m: {"attributions": r["attributions"].tolist(), **{x: v for x, v in r.items() if x != "attributions"}}
               for m, r in methods.items()}}


def summarise(per_ex: list[dict], cfg: dict) -> dict:
    faith = {m: {} for m in METHODS}
    for k in map(str, cfg["deletion_k"]):
        for m in METHODS:
            faith[m][k] = {"top_drop": float(np.mean([e["faithfulness"][k][m]["top_drop"] for e in per_ex])),
                           "top_flip": float(np.mean([e["faithfulness"][k][m]["top_flip"] for e in per_ex])),
                           "rand_drop": float(np.mean([e["faithfulness"][k]["rand_drop"] for e in per_ex])),
                           "rand_flip": float(np.mean([e["faithfulness"][k]["rand_flip"] for e in per_ex]))}
    taus = [e["kendall_tau"] for e in per_ex if e["kendall_tau"] is not None]
    return {"faithfulness": faith,
            "agreement": {"kendall_tau_mean": float(np.mean(taus)), "kendall_tau_sd": float(np.std(taus)),
                          "top3_overlap_mean": float(np.mean([e["top3_overlap"] for e in per_ex]))},
            "evaluations_per_explanation": {m: float(np.mean([e[m]["calls"] for e in per_ex])) for m in METHODS},
            "shap_additivity_error": {"mean": float(np.mean([e["kernel_shap"]["additivity_error"] for e in per_ex])),
                                      "max": float(np.max([e["kernel_shap"]["additivity_error"] for e in per_ex]))},
            "lime_r2_mean": float(np.mean([e["lime"]["r2"] for e in per_ex])),
            "mean_units": float(np.mean([e["n_units"] for e in per_ex]))}


def figures(ds: Dataset, res: dict, mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, e in zip(axes, res["examples"][:2]):
        s, w = np.array(e["kernel_shap"]["attributions"]), np.array(e["lime"]["attributions"])
        top = np.argsort(-np.abs(s))[:8][::-1]
        yy = np.arange(len(top))
        ax.barh(yy + 0.2, s[top], height=0.4, label="KernelSHAP")
        ax.barh(yy - 0.2, w[top], height=0.4, label="LIME")
        ax.set_yticks(yy, [e["units"][j][:28] for j in top], fontsize=7)
        ax.axvline(0, c="k", lw=0.5)
        ax.set(title=f"{e['id']}: P({e['predicted'][:18]}) = {e['p_full']:.2f}", xlabel="attribution")
        ax.title.set_fontsize(8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    savefig(fig, mode, ds.name, STAGE, "attributions")

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ks, f = [int(k) for k in res["faithfulness"]["kernel_shap"]], res["faithfulness"]
    for m, col, label in (("kernel_shap", "C0", "KernelSHAP top-k"), ("lime", "C1", "LIME top-k")):
        ax.plot(ks, [f[m][str(k)]["top_drop"] for k in ks], "o-", c=col, ms=4, label=label)
    ax.plot(ks, [f["kernel_shap"][str(k)]["rand_drop"] for k in ks], "s--", c="grey", ms=4, label="random k")
    ax.set(xlabel="units deleted (k)", ylabel="drop in predicted-class probability", xticks=ks, title=ds.title)
    ax.legend(fontsize=7)
    savefig(fig, mode, ds.name, STAGE, "faithfulness")


def run_dataset(ds: Dataset, mode: str, cb: CachedBackend, cfg: dict) -> dict:
    examples, rule = choose_examples(ds, cfg)
    ledger_before = Ledger().count(STAGE) if mode == "live" else 0
    per_ex = [explain_example(ProbabilityFunction(cb, ds, ex), cfg, i) for i, ex in enumerate(examples)]
    new_requests = (Ledger().count(STAGE) - ledger_before) if mode == "live" else 0
    cost = exchange_cost(ds.name) if mode == "live" else None
    requests = cost["logged_requests"] if cost else 0       # every request ever sent for this dataset, from the request log
    empty = None
    if ds.binary:   # P(positive class | every unit removed), read off the KernelSHAP base values
        vals = [e["kernel_shap"]["base_value"] if e["predicted"] == ds.positive else 1 - e["kernel_shap"]["base_value"] for e in per_ex]
        empty = float(np.mean(vals))
    res = {"dataset": ds.name, "title": ds.title, "model": cb.backend.model_version, "n_examples": len(examples),
           "positive_given_empty_input": empty,
           "selection": rule, "unit": "feature" if ds.kind == "tabular" else "word", **summarise(per_ex, cfg),
           "jev_requests": requests, "jev_requests_this_run": new_requests,
           "failed_requests": cost["failed_requests"] if cost else 0, "examples": per_ex}
    if mode == "live":
        res["cost_usd"] = cost
    save_json(mode, ds.name, STAGE, "explain", res, [cb.backend.mode])
    figures(ds, res, mode)
    return res


def exchange_cost(dataset: str) -> dict:
    """Requests, input tokens and cost of this dataset's explain requests, from the request log."""
    import json

    from ..backends import exchange

    path, tokens, n, failed = exchange.log_dir() / f"{STAGE}.jsonl", 0, 0, 0
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("dataset") != dataset:
                    continue
                n += 1
                if r.get("status") == 200 and isinstance(r.get("response"), dict):
                    tokens += int(r["response"].get("usage", {}).get("input_tokens", 0))
                else:
                    failed += 1
    return {"logged_requests": n, "failed_requests": failed, "input_tokens": tokens,
            "usd": tokens * config.load("jev")["usd_per_mtok_input"] / 1e6}


def redraw(mode: str, names: list[str]) -> None:
    """Draw the figures again from the saved results: no model call, no recomputation."""
    from ..data import datasets
    from .common import read_json
    for name in names:
        figures(datasets.load(name), read_json(mode, name, STAGE, "explain"), mode)


def run(args) -> int:
    cfg = config.load("explain")
    names = selected(args)
    budget = Budget(STAGE, dry_run=args.dry_run)
    plan, worst = [], 0
    for name in names:
        ds = datasets.load(name)
        n = len(choose_examples(ds, cfg)[0])
        worst += n * requests_per_example(cfg)
        plan.append(ds)
    if args.mode == "live":   # nothing is sent unless the worst case fits the stage budget and the cap
        print(f"explain: worst case {worst} Jev requests for {len(plan)} dataset(s) "
              f"({requests_per_example(cfg)} per example); cached inputs cost nothing")
        budget.preflight({"jev": worst})
    else:
        budget.preflight({})
    for ds in plan:
        cb = CachedBackend(backends.get("jev", args.mode, ds, stage=STAGE), budget=budget)
        res = run_dataset(ds, args.mode, cb, cfg)
        f3 = {m: res["faithfulness"][m]["3"] for m in METHODS}
        print(f"explain ({args.mode}): {ds.name}: {res['n_examples']} examples; k=3 probability drop KernelSHAP "
              f"{f3['kernel_shap']['top_drop']:.3f}, LIME {f3['lime']['top_drop']:.3f}, random {f3['lime']['rand_drop']:.3f}; "
              f"tau {res['agreement']['kendall_tau_mean']:.2f}; additivity error max {res['shap_additivity_error']['max']:.1e}; "
              f"Jev requests {res['jev_requests']} in total, {res['jev_requests_this_run']} in this run")
    return 0
