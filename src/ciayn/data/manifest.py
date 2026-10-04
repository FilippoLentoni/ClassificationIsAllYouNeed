"""Frozen evaluation identifiers (design D2).

`freeze` writes one split file per dataset and `MANIFEST.sha256`. Every other stage loads
datasets through `datasets.load()`, which calls `verify()`: the split file must exist, its
hash must equal the manifest entry, and the splits computed from configuration must equal
the file. Any difference stops the stage before it does work.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .. import config
from .datasets import Dataset

MANIFEST = "MANIFEST.sha256"


class ManifestError(RuntimeError):
    pass


def split_payload(ds: Dataset) -> dict:
    return {"dataset": ds.name, "title": ds.title, "protocol": ds.protocol, "classes": ds.classes,
            "positive": ds.positive, "source": ds.source,
            "sizes": [{k: len(v) for k, v in r.items()} for r in ds.repeats],
            "n_evaluation": len(ds.eval_ids), "n_jev": len(ds.jev_ids),
            "jev_val": ds.jev_val, "repeats": ds.repeats}


def _dump(payload: dict) -> bytes:
    return json.dumps(payload, indent=None, separators=(",", ":"), sort_keys=True).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest() -> dict[str, str]:
    path = config.SPLITS / MANIFEST
    if not path.exists():
        raise ManifestError(f"{path} is missing: run `python -m ciayn freeze` before any other stage")
    return {name: digest for digest, name in (ln.split() for ln in path.read_text().splitlines()
                                              if ln and not ln.startswith("#"))}


def freeze(dss: list[Dataset], overwrite: bool = False) -> dict[str, str]:
    """Write the split files and the manifest. Refuses to replace an existing freeze unless told to."""
    config.SPLITS.mkdir(parents=True, exist_ok=True)
    if (config.SPLITS / MANIFEST).exists() and not overwrite:
        raise ManifestError(f"{config.SPLITS / MANIFEST} already exists; evaluation identifiers are frozen. "
                            "Pass --refreeze only if nothing has used them yet.")
    digests = {}
    for ds in dss:
        data = _dump(split_payload(ds))
        (config.SPLITS / f"{ds.name}.json").write_bytes(data)
        digests[f"{ds.name}.json"] = sha256(data)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (config.SPLITS / MANIFEST).write_text(f"# frozen {stamp}\n" + "".join(f"{d}  {n}\n" for n, d in sorted(digests.items())))
    return digests


def remove_dataset(name: str) -> dict:
    """Take one dataset out of the frozen set without touching the others.

    The manifest keeps its `# frozen` header and the other lines byte for byte; only the removed
    dataset's line and split file go. The removal is recorded with the unchanged hashes. Refuses
    unless what was spent on the dataset has been saved first.
    """
    from ..paper import provenance
    saved = provenance.removed_path()
    if not saved.exists() or name not in json.loads(saved.read_text()).get("datasets", {}):
        raise ManifestError(f"what was spent on {name} has not been recorded: run "
                            f"`python -m ciayn provenance --mode live --removed {name}` first")
    path = config.SPLITS / MANIFEST
    lines = path.read_text().splitlines(keepends=True)
    gone = [ln for ln in lines if ln.split()[-1:] == [f"{name}.json"]]
    if len(gone) != 1:
        raise ManifestError(f"{name} is not in the manifest (or appears more than once)")
    before = read_manifest()
    path.write_text("".join(ln for ln in lines if ln not in gone))
    (config.SPLITS / f"{name}.json").unlink(missing_ok=True)
    after = read_manifest()
    if after != {k: v for k, v in before.items() if k != f"{name}.json"}:
        raise ManifestError("the manifest entries of the remaining datasets changed")
    rec_path = config.SPLITS / "REMOVED.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"removed": []}
    rec["removed"].append({"dataset": name, "removed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "removed_sha256": before[f"{name}.json"], "frozen_at": frozen_at(), "unchanged_sha256": after})
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec["removed"][-1]


def frozen_at() -> str:
    first = (config.SPLITS / MANIFEST).read_text().splitlines()[0]
    return first.removeprefix("# frozen ").strip()


def verify(ds: Dataset) -> None:
    name = f"{ds.name}.json"
    recorded = read_manifest().get(name)
    path = config.SPLITS / name
    if recorded is None or not path.exists():
        raise ManifestError(f"{ds.name} has no frozen split file in {config.SPLITS}: run `python -m ciayn freeze`")
    data = path.read_bytes()
    if sha256(data) != recorded:
        raise ManifestError(f"{path} does not match its hash in {MANIFEST}: the frozen identifiers were changed")
    frozen = json.loads(data)
    if frozen["repeats"] != ds.repeats or frozen["jev_val"] != ds.jev_val:
        raise ManifestError(f"the splits computed for {ds.name} differ from the frozen ones in {path}: "
                            "configuration or code changed after freezing")
