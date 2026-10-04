"""Shared plumbing for the offline experiments: mode-labelled artifacts and the bootstrap."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Embed TrueType fonts in figures: the NeurIPS format does not accept Type 3 fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np  # noqa: E402

from .. import config  # noqa: E402

MODEL_LABEL = {"jev": "Jev (zero-shot)", "qwen": "Qwen3.6-27B (zero-shot)", "xgb": "XGBoost (supervised)",
               "bert": "BERT (supervised)", "qwen_think": "Qwen3.6-27B, thinking (zero-shot)"}
MODEL_COLOUR = {"jev": "C0", "xgb": "C1", "qwen": "C2", "bert": "C3", "qwen_think": "C4"}


def selected(args) -> list[str]:
    name = getattr(args, "dataset", "all")
    return list(config.DATASETS) if name in (None, "all") else [name]


def out_dir(mode: str, dataset: str, stage: str):
    return config.results_dir(mode, dataset, stage)


def save_json(mode: str, dataset: str, stage: str, name: str, obj: dict, artifact_modes: list[str] | None = None) -> None:
    """Every artifact carries the mode it was produced in (mock results must never pass as live)."""
    payload = {"mode": mode, "artifact_modes": sorted(set(artifact_modes or [mode])), **obj}
    (out_dir(mode, dataset, stage) / f"{name}.json").write_text(json.dumps(payload, indent=1, default=_jsonable))


def read_json(mode: str, dataset: str, stage: str, name: str) -> dict:
    path = config.RESULTS / mode / dataset / stage / f"{name}.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run `python -m ciayn {stage} --mode {mode}` first")
    return json.loads(path.read_text())


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def savefig(fig, mode: str, dataset: str, stage: str, name: str) -> None:
    fig.tight_layout()
    if mode == "mock":
        fig.text(0.5, 0.5, "MOCK RESULTS", fontsize=28, color="red", alpha=0.25, ha="center", va="center",
                 rotation=30, transform=fig.transFigure)
    # no creation date: reruns on the same inputs give byte-identical figures
    fig.savefig(out_dir(mode, dataset, stage) / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def bootstrap_ci(stat, arrays: tuple[np.ndarray, ...], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of stat(*arrays) over items (fixed seed)."""
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = stat(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mean_sd(values) -> dict:
    v = np.asarray(list(values), float)
    return {"value": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_repeats": int(len(v)),
            "values": v.tolist()}
