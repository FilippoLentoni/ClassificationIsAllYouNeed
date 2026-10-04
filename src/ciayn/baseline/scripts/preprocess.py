"""SageMaker processing job: turn each repeat's examples into feature matrices.

Standalone (no project imports): it runs inside the scikit-learn container. For every
repeat directory it fits the preprocessing on the training examples only and applies it
unchanged to the validation and test examples. Test examples arrive without labels.

Input  (/opt/ml/processing/input/data):  job.json, repeat_<r>/{train,val,test}.csv
Output (/opt/ml/processing/output):      job.json, preprocess.json, repeat_<r>/{X_*.npz, y_*.npy, ids_*.json}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def fit_tabular(train: pd.DataFrame, job: dict):
    """One-hot encoding of categorical columns fitted on train; numeric columns pass through."""
    from sklearn.preprocessing import OneHotEncoder

    cat, num = job["categorical"], job["numeric"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32) if cat else None
    if enc is not None:
        enc.fit(train[cat].astype(str))

    def transform(df: pd.DataFrame) -> sp.csr_matrix:
        parts = []
        if num:
            parts.append(sp.csr_matrix(df[num].to_numpy(dtype=np.float32)))
        if enc is not None:
            parts.append(enc.transform(df[cat].astype(str)))
        return sp.hstack(parts, format="csr")

    names = list(num) + ([f"{c}={v}" for c, vals in zip(cat, enc.categories_) for v in vals] if enc is not None else [])
    return transform, names


def fit_text(train: pd.DataFrame, job: dict):
    """TF-IDF on unigrams and bigrams, vocabulary and weights fitted on train."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=tuple(job["ngram_range"]), max_features=job["max_features"],
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(train["text"].fillna(""))
    return (lambda df: vec.transform(df["text"].fillna("")).tocsr()), list(vec.get_feature_names_out())


def process_repeat(src: Path, dst: Path, job: dict) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(src / f"{s}.csv", keep_default_na=False, na_values=[""]) for s in ("train", "val", "test")}
    if "y" in frames["test"].columns:
        raise ValueError(f"{src}/test.csv must not contain labels")
    transform, names = (fit_tabular if job["kind"] == "tabular" else fit_text)(frames["train"], job)  # train only
    info = {"n_features": len(names)}
    for split, df in frames.items():
        X = transform(df)
        sp.save_npz(dst / f"X_{split}.npz", X)
        (dst / f"ids_{split}.json").write_text(json.dumps(df["id"].tolist()))
        if split != "test":
            np.save(dst / f"y_{split}.npy", df["y"].to_numpy(dtype=np.int64))
        info[f"n_{split}"] = int(X.shape[0])
    return info


def main(src: Path, dst: Path) -> dict:
    import sklearn

    job = json.loads((src / "job.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    repeats = {}
    for r in range(job["n_repeats"]):
        repeats[str(r)] = process_repeat(src / f"repeat_{r}", dst / f"repeat_{r}", job)
        print(f"preprocess: {job['dataset']} repeat {r}: {repeats[str(r)]}", flush=True)
    (dst / "job.json").write_text(json.dumps(job))
    record = {"dataset": job["dataset"], "repeats": repeats,
              "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__}}
    (dst / "preprocess.json").write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    base = Path(os.environ.get("CIAYN_PROCESSING_DIR", "/opt/ml/processing"))
    main(base / "input" / "data", base / "output")
