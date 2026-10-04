"""Prediction files for every (dataset, model) and the score tables the experiments read.

`results/<mode>/<dataset>/jev/predictions.parquet` has one row per example Jev was queried on.
`results/<mode>/<dataset>/{xgb,bert}/predictions.parquet` have one row per (repeat, split, example),
because a supervised model is a different fitted model in every repeat. `qwen` and `qwen_think`
have one row per item, like Jev.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .backends.base import INVALID, Prediction
from .data.datasets import Dataset


def model_dir(mode: str, dataset: str, model: str):
    return config.results_dir(mode, dataset, model)


def pred_path(mode: str, dataset: str, model: str):
    return model_dir(mode, dataset, model) / "predictions.parquet"


def prob_columns(k: int) -> list[str]:
    return [f"p{i}" for i in range(k)]


def write_jev_predictions(mode: str, ds: Dataset, rows: list[tuple[str, Prediction]], name: str = "predictions") -> pd.DataFrame:
    """Jev's predictions. `name` "predictions" is the frozen item set; "escalation_val" holds the
    validation items predicted only for the escalation threshold (AG News)."""
    recs = []
    for ex_id, p in rows:
        rec = {"id": ex_id, "y": int(ds.y[ex_id]), "label": p.label, "valid": p.valid,
               "missing": bool(p.meta.get("missing", False)),
               "latency_s": float(p.meta.get("latency_s", np.nan)), "cost_usd": float(p.meta.get("cost_usd", np.nan)),
               "input_tokens": float(p.meta.get("input_tokens") or np.nan),
               "model_version": str(p.meta.get("model_version", "")), "mode": str(p.meta.get("mode", mode)),
               "jev_confidence": float(p.meta.get("jev_confidence", np.nan))}
        rec.update({f"p{k}": float(v) for k, v in enumerate(p.probs)})
        recs.append(rec)
    df = pd.DataFrame(recs)
    if name != "predictions":
        df.to_parquet(model_dir(mode, ds.name, "jev") / f"{name}.parquet", index=False)
        return df
    df.to_parquet(pred_path(mode, ds.name, "jev"), index=False)
    _summary(mode, ds, "jev", df, {"invalid": int((~df.valid).sum()), "missing": int(df.missing.sum()),
                                   "total_cost_usd": float(np.nansum(df.cost_usd)),
                                   "total_input_tokens": float(np.nansum(df.input_tokens)),
                                   "mean_latency_s": float(np.nanmean(df.latency_s))})
    return df


def write_xgb_predictions(mode: str, ds: Dataset, frame: pd.DataFrame, run: dict) -> pd.DataFrame:
    return write_supervised_predictions(mode, ds, "xgb", frame, run)


def write_supervised_predictions(mode: str, ds: Dataset, model: str, frame: pd.DataFrame, run: dict) -> pd.DataFrame:
    """`frame`: id, repeat, split, p0..p{K-1}. Gold labels are joined here; they are never part of the input."""
    df = frame.copy()
    cols = prob_columns(len(ds.classes))
    P = df[cols].to_numpy()
    df["y"] = ds.y.loc[df["id"]].to_numpy()
    df["label"] = P.argmax(1)
    df["valid"] = np.isfinite(P).all(1)
    df["mode"] = mode
    df["model_version"] = str(run.get("model_version", ""))
    df = df[["id", "repeat", "split", "y", "label", "valid", "mode", "model_version"] + cols]
    df.to_parquet(pred_path(mode, ds.name, model), index=False)
    _summary(mode, ds, model, df, {"invalid": int((~df.valid).sum())})
    (model_dir(mode, ds.name, model) / "run.json").write_text(json.dumps({"mode": mode, **run}, indent=1))
    return df


def write_item_predictions(mode: str, ds: Dataset, model: str, frame: pd.DataFrame, run: dict) -> pd.DataFrame:
    """Zero-shot predictions other than Jev's (`qwen`, `qwen_think`): one row per item.

    `frame`: id, valid, p0..p{K-1} and optional timing/token columns. Invalid rows get a uniform
    distribution and the INVALID label. Gold labels are joined here.
    """
    df = frame.copy()
    cols = prob_columns(len(ds.classes))
    valid = df["valid"].to_numpy(bool) & np.isfinite(df[cols].to_numpy()).all(1)
    df.loc[~valid, cols] = 1.0 / len(cols)
    df["valid"] = valid
    df["y"] = ds.y.loc[df["id"]].to_numpy()
    df["label"] = np.where(valid, df[cols].to_numpy().argmax(1), INVALID)
    df["mode"] = mode
    df["model_version"] = str(run.get("model_version", ""))
    extra = [c for c in df.columns if c not in {"id", "y", "label", "valid", "mode", "model_version", *cols}]
    df = df[["id", "y", "label", "valid", "mode", "model_version"] + cols + extra]
    df.to_parquet(pred_path(mode, ds.name, model), index=False)
    _summary(mode, ds, model, df, {"invalid": int((~df.valid).sum())})
    (model_dir(mode, ds.name, model) / "run.json").write_text(json.dumps({"mode": mode, **run}, indent=1, default=str))
    return df


def _summary(mode: str, ds: Dataset, model: str, df: pd.DataFrame, extra: dict) -> None:
    summary = {"dataset": ds.name, "model": model, "mode": mode,
               "artifact_modes": sorted(df["mode"].unique().tolist()),
               "model_version": sorted(df["model_version"].unique().tolist()), "n_rows": int(len(df)), **extra}
    (model_dir(mode, ds.name, model) / "predictions.json").write_text(json.dumps(summary, indent=1))


def read_predictions(mode: str, dataset: str, model: str) -> pd.DataFrame:
    path = pred_path(mode, dataset, model)
    if not path.exists():
        how = ("predict" if model == "jev" or mode == "mock" else
               {"xgb": "baseline fetch", "bert": "encoder fetch"}.get(model, "llm fetch"))
        raise RuntimeError(f"missing {path}; run `python -m ciayn {how} --mode {mode}` first")
    return pd.read_parquet(path)


def read_summary(mode: str, dataset: str, model: str) -> dict:
    return json.loads((model_dir(mode, dataset, model) / "predictions.json").read_text())


@dataclass
class Scores:
    """A model's predictions on one split of one repeat, in the split's identifier order."""
    ids: list[str]
    y: np.ndarray        # gold class index
    P: np.ndarray        # n x K probabilities (uniform for invalid/missing items)
    pred: np.ndarray     # argmax label, INVALID for invalid/missing items
    valid: np.ndarray    # bool
    modes: list[str]

    @property
    def conf(self) -> np.ndarray:
        return self.P.max(1)

    def __len__(self) -> int:
        return len(self.ids)


def _to_scores(d: pd.DataFrame, ids: list[str], k: int) -> Scores:
    d = d.set_index("id")
    missing = [i for i in ids if i not in d.index]
    if missing:
        raise RuntimeError(f"predictions are missing {len(missing)} identifiers, for example {missing[:3]}")
    d = d.loc[ids]
    valid = d.valid.to_numpy(bool)
    pred = np.where(valid, d.label.to_numpy(int), INVALID)
    return Scores(list(ids), d.y.to_numpy(int), d[prob_columns(k)].to_numpy(float), pred, valid,
                  sorted(d["mode"].unique().tolist()))


def scores(mode: str, ds: Dataset, model: str, split: str = "test", repeat: int = 0,
           frame: pd.DataFrame | None = None) -> Scores:
    """Predictions of `model` on a split of a repeat. Jev's validation part is its own validation items."""
    df = read_predictions(mode, ds.name, model) if frame is None else frame
    k = len(ds.classes)
    if model in config.ITEM_MODELS:
        return _to_scores(df, ds.jev_split(split, repeat), k)
    part = df[(df.repeat == repeat) & (df.split == split)]
    return _to_scores(part, ds.ids(split, repeat), k)


def same_examples(a: Scores, b: Scores, what: str) -> None:
    if a.ids != b.ids:
        raise RuntimeError(f"{what}: the models were not scored on the same examples "
                           f"({len(a.ids)} vs {len(b.ids)} identifiers)")


def subset(s: Scores, ids: list[str]) -> Scores:
    """The rows of `s` for `ids`, in that order; fails if any identifier is absent."""
    pos = {i: n for n, i in enumerate(s.ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} identifiers are not among the scored examples, for example {missing[:3]}")
    idx = np.array([pos[i] for i in ids], int)
    return Scores(list(ids), s.y[idx], s.P[idx], s.pred[idx], s.valid[idx], s.modes)
