import json

import numpy as np
import pytest

from ciayn import config
from ciayn.data import datasets, manifest

HELDOUT = {"sst2": (872, 2000, 65349, 2), "agnews": (7600, 5000, 115000, 4)}
REPEATED = {"german": (1000, 600, 200, 200), "wdbc": (569, 341, 114, 114)}


@pytest.mark.parametrize("name", list(HELDOUT))
def test_official_heldout_split(name):
    ds = datasets.load(name)
    n_eval, n_val, n_train, k = HELDOUT[name]
    r = ds.repeats[0]
    assert len(ds.repeats) == 1 and (len(r["test"]), len(r["val"]), len(r["train"])) == (n_eval, n_val, n_train)
    assert len(ds.classes) == k and ds.eval_ids == r["test"]
    held = set(r["test"])
    assert all(i.startswith(f"{name}-train-") for i in r["train"] + r["val"]), "train and val come from the official training split"
    assert not held & set(r["train"]) and not held & set(r["val"]) and not set(r["train"]) & set(r["val"])
    # stratified validation draw
    full = np.bincount(ds.labels(r["train"] + r["val"]), minlength=k) / (n_train + n_val)
    part = np.bincount(ds.labels(r["val"]), minlength=k) / n_val
    assert np.abs(full - part).max() < 0.01


@pytest.mark.parametrize("name", list(REPEATED))
def test_repeated_splits_sizes_disjoint_and_stratified(name):
    ds = datasets.load(name)
    n, n_train, n_val, n_test = REPEATED[name]
    assert len(ds.frame) == n and len(ds.repeats) == 10 and len(ds.jev_ids) == n
    base = np.mean(ds.y.to_numpy())
    for r in ds.repeats:
        assert (len(r["train"]), len(r["val"]), len(r["test"])) == (n_train, n_val, n_test)
        ids = r["train"] + r["val"] + r["test"]
        assert len(set(ids)) == n, "train, val and test are disjoint and cover every row"
        for part in ("train", "val", "test"):
            assert abs(np.mean(ds.labels(r[part])) - base) < 0.01
    assert ds.repeats[0]["test"] != ds.repeats[1]["test"]


@pytest.mark.parametrize("name", list(REPEATED) + list(HELDOUT))
def test_splits_are_deterministic(name):
    a = datasets.build(name).repeats
    datasets.clear_cache()
    b = datasets.build(name)
    assert a == b.repeats and b.jev_val == datasets.build(name).jev_val


def test_class_order_and_positive_class():
    assert datasets.load("german").classes == ["good", "bad"] and datasets.load("german").positive == "bad"
    assert datasets.load("wdbc").classes == ["benign", "malignant"] and datasets.load("wdbc").positive == "malignant"
    assert datasets.load("sst2").classes == ["negative", "positive"] and datasets.load("sst2").positive == "positive"
    assert datasets.load("agnews").classes == ["World", "Sports", "Business", "Sci/Tech"]
    a = datasets.load("agnews")
    assert a.positive is None and list(a.criteria) == a.classes, "descriptions are in class order"


def test_wdbc_labels_match_the_source_counts():
    ds = datasets.load("wdbc")
    assert np.bincount(ds.y.to_numpy()).tolist() == [357, 212]   # benign, malignant


def test_sst2_jev_validation_items():
    ds = datasets.load("sst2")
    assert len(ds.jev_val) == 500 and ds.jev_val == ds.repeats[0]["val"][:500]
    assert len(ds.jev_ids) == 872 + 500 and ds.jev_split("val") == ds.jev_val
    assert datasets.load("agnews").jev_val == [] and len(datasets.load("agnews").jev_ids) == 7600


# ---- frozen identifiers ---------------------------------------------------------------------

@pytest.fixture
def frozen_copy(tmp_path, monkeypatch):
    for p in config.SPLITS.iterdir():
        (tmp_path / p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(config, "SPLITS", tmp_path)
    datasets.clear_cache()
    yield tmp_path
    datasets.clear_cache()


def test_frozen_files_match_their_manifest():
    recorded = manifest.read_manifest()
    assert set(recorded) == {f"{n}.json" for n in config.DATASETS}
    for name, digest in recorded.items():
        assert manifest.sha256((config.SPLITS / name).read_bytes()) == digest


def test_changed_split_file_stops_loading(frozen_copy):
    path = frozen_copy / "wdbc.json"
    d = json.loads(path.read_text())
    d["repeats"][0]["test"][0], d["repeats"][0]["train"][0] = d["repeats"][0]["train"][0], d["repeats"][0]["test"][0]
    path.write_text(json.dumps(d))
    with pytest.raises(manifest.ManifestError, match="does not match its hash"):
        datasets.load("wdbc")


def test_splits_that_differ_from_the_frozen_ones_stop_loading(frozen_copy, monkeypatch):
    monkeypatch.setitem(config.load("datasets"), "repeats", 9)   # configuration changed after freezing
    with pytest.raises(manifest.ManifestError, match="differ from the frozen ones"):
        datasets.load("german")


def test_missing_manifest_stops_loading_and_freeze_refuses_to_overwrite(frozen_copy):
    with pytest.raises(manifest.ManifestError, match="already exists"):
        manifest.freeze([datasets.build("wdbc")])
    (frozen_copy / manifest.MANIFEST).unlink()
    datasets.clear_cache()
    with pytest.raises(manifest.ManifestError, match="run `python -m ciayn freeze`"):
        datasets.load("wdbc")


def test_dataset_documents_contain_every_required_item():
    for name in config.DATASETS:
        text = (config.DOCS / "datasets" / f"{name}.md").read_text()
        ds = datasets.load(name)
        for item in ("**Source and version:**", "**Positive class:**", "**Split protocol:**", "**Identifiers frozen:**",
                     "## Preprocessing and serialization", "**Missing values", "**Instructions:**",
                     "Options and descriptions, in the order sent"):
            assert item in text, (name, item)
        assert all(f"`{c}`" in text for c in ds.classes)
        assert ("seeds 0 to 9" in text) if ds.protocol == "repeated" else ("with seed 0" in text)
        assert manifest.read_manifest()[f"{name}.json"] in text
