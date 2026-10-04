"""`remove-dataset` stage: take a dataset out of the study after work was done on it.

It removes the dataset's frozen split file and manifest entry (the other datasets' files and
hashes are left byte-identical) and deletes its saved results in both modes. It refuses to run
until `provenance --removed <dataset>` has saved what was spent on the dataset, because that
cannot be recomputed from tracked files afterwards. The response cache, ledger and request log
are not touched, so every total still includes the removed dataset.
"""
from __future__ import annotations

import shutil

from . import config
from .data import manifest


def run(args) -> int:
    name = args.removed
    if not name:
        raise RuntimeError("remove-dataset needs --removed <dataset>")
    if name in config.DATASETS:
        raise RuntimeError(f"{name} is still in config.DATASETS; remove it from the configuration first")
    rec = manifest.remove_dataset(name)
    deleted = []
    for mode in config.MODES:
        d = config.RESULTS / mode / name
        if d.exists():
            shutil.rmtree(d)
            deleted.append(str(d.relative_to(config.ROOT)) if d.is_relative_to(config.ROOT) else str(d))
    print(f"remove-dataset: {name}: manifest entry and split file removed; {len(rec['unchanged_sha256'])} remaining "
          f"datasets unchanged; deleted {deleted or 'no results directories'}")
    return 0
