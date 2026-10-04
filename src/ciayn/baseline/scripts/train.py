"""SageMaker training job: XGBoost for every repeat, with validation-only model selection.

Standalone (no project imports): it runs inside the XGBoost container. For each repeat it
tries a fixed hyperparameter grid with early stopping on the validation part, keeps the
configuration with the lowest validation log-loss, and writes class probabilities for the
validation and test examples. It then refits the chosen configuration on nested, stratified
fractions of the training part for the learning curve. Test labels are never read: they are
not in the input.

Input  (/opt/ml/input/data/features):  job.json, repeat_<r>/{X_*.npz, y_{train,val}.npy, ids_*.json}
Output (/opt/ml/output/data):          predictions.csv, learning_curve.csv, run.json
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xgboost as xgb


START = time.time()


def params(k: int, max_depth: int, eta: float, seed: int, max_bin: int = 256) -> dict:
    p = {"max_depth": max_depth, "eta": eta, "tree_method": "hist", "max_bin": max_bin, "seed": seed, "verbosity": 0,
         "nthread": os.cpu_count() or 1}
    p.update({"objective": "binary:logistic", "eval_metric": "logloss"} if k == 2 else
             {"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss"})
    return p


def proba(booster, dm, k: int) -> np.ndarray:
    p = booster.predict(dm, iteration_range=(0, booster.best_iteration + 1))
    return np.column_stack([1 - p, p]) if k == 2 else p.reshape(-1, k)


def fit(Xtr, ytr, dval, k, max_depth, eta, seed, job):
    max_bin = int(job.get("max_bin", 256))
    booster = xgb.train(params(k, max_depth, eta, seed, max_bin), xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=max_bin),
                        num_boost_round=job["max_rounds"], evals=[(dval, "val")], early_stopping_rounds=job["patience"],
                        verbose_eval=False)
    return booster, float(booster.best_score)


def nested_subset(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Stratified subset; the subset for a smaller fraction is contained in the one for a larger fraction."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        keep.append(idx[: max(1, int(round(fraction * len(idx))))])
    return np.sort(np.concatenate(keep))


def run_repeat(src: Path, r: int, job: dict) -> tuple[list, list, dict]:
    k = job["n_classes"]
    X = {s: sp.load_npz(src / f"X_{s}.npz").tocsr() for s in ("train", "val", "test")}
    ids = {s: json.loads((src / f"ids_{s}.json").read_text()) for s in ("val", "test")}
    ytr, yval = np.load(src / "y_train.npy"), np.load(src / "y_val.npy")
    dval, dtest = xgb.DMatrix(X["val"], label=yval), xgb.DMatrix(X["test"])
    grid, best = [], None
    for max_depth, eta in itertools.product(job["grid"]["max_depth"], job["grid"]["eta"]):
        t0 = time.time()
        booster, loss = fit(X["train"], ytr, dval, k, max_depth, eta, r, job)   # selection uses validation only
        grid.append({"max_depth": max_depth, "eta": eta, "val_logloss": loss, "best_iteration": int(booster.best_iteration),
                     "seconds": round(time.time() - t0, 1)})
        print(f"train: repeat {r} grid {grid[-1]}", flush=True)
        if best is None or loss < best[1]:
            best = (booster, loss, max_depth, eta)
    booster, loss, max_depth, eta = best
    preds = [(i, r, "val", *row) for i, row in zip(ids["val"], proba(booster, dval, k))]
    t_infer = time.time()
    full_test = proba(booster, dtest, k)                 # timed: batch inference time per evaluation item
    infer_s = time.time() - t_infer
    preds += [(i, r, "test", *row) for i, row in zip(ids["test"], full_test)]
    curve, sizes, skipped = [], {}, {}
    for frac in job["fractions"]:
        if frac >= 1.0:
            p_test, n = full_test, len(ytr)
        else:
            sub = nested_subset(ytr, frac, job["seed"] + r)
            if len(np.unique(ytr[sub])) < k:            # a class is absent at this size: skip rather than mis-shape
                skipped[str(frac)] = "a class has no training example at this size"
                continue
            if time.time() - START > job.get("time_budget_s", float("inf")):
                skipped[str(frac)] = "the job's time budget for learning-curve fits was used up"
                continue
            b, _ = fit(X["train"][sub], ytr[sub], dval, k, max_depth, eta, r, job)
            p_test, n = proba(b, dtest, k), len(sub)
            print(f"train: repeat {r} learning curve fraction {frac} ({n} labels) done", flush=True)
        sizes[str(frac)] = int(n)
        curve += [(i, r, frac, n, *row) for i, row in zip(ids["test"], p_test)]
    record = {"selected": {"max_depth": max_depth, "eta": eta, "val_logloss": loss, "best_iteration": int(booster.best_iteration)},
              "grid": grid, "n_features": int(X["train"].shape[1]), "curve_train_sizes": sizes, "curve_skipped": skipped,
              "infer_test_s": infer_s, "infer_s_per_item": infer_s / max(len(ids["test"]), 1),
              "label_budget": {"training": int(len(ytr)), "early_stopping_and_selection": int(len(yval)),
                               "threshold_selection": int(len(yval)) if k == 2 else 0, "calibration": 0}}
    return preds, curve, record


def write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) if not isinstance(v, (float, np.floating)) else f"{v:.6g}" for v in row) + "\n")


def main(src: Path, dst: Path) -> dict:
    job = json.loads((src / "job.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    cols = [f"p{i}" for i in range(job["n_classes"])]
    preds, curve, repeats, t0 = [], [], {}, time.time()
    for r in range(job["n_repeats"]):
        p, c, rec = run_repeat(src / f"repeat_{r}", r, job)
        preds += p
        curve += c
        repeats[str(r)] = rec
        print(f"train: {job['dataset']} repeat {r}: {rec['selected']} ({time.time() - t0:.0f}s elapsed)", flush=True)
    write_csv(dst / "predictions.csv", ["id", "repeat", "split"] + cols, preds)
    write_csv(dst / "learning_curve.csv", ["id", "repeat", "fraction", "n_train"] + cols, curve)
    record = {"dataset": job["dataset"], "repeats": repeats, "grid_searched": job["grid"], "max_rounds": job["max_rounds"],
              "max_bin": int(job.get("max_bin", 256)),
              "patience": job["patience"], "fractions": job["fractions"], "seconds": round(time.time() - t0, 1),
              "versions": {"python": sys.version.split()[0], "xgboost": xgb.__version__, "numpy": np.__version__}}
    (dst / "run.json").write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    base = Path(os.environ.get("CIAYN_TRAINING_DIR", "/opt/ml"))
    main(base / "input" / "data" / "features", base / "output" / "data")
