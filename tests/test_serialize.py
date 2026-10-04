import pytest

from ciayn import config
from ciayn.data import datasets
from ciayn.data.serialize import MISSING_VALUE, serialize, units


@pytest.mark.parametrize("name", ["german", "wdbc"])
def test_tabular_serialization(name):
    ds = datasets.load(name)
    for ex_id in ds.jev_ids[:300]:
        text = serialize(ds, ex_id)
        lines = text.splitlines()
        assert [ln.split(": ", 1)[0] for ln in lines] == ds.features, "every feature exactly once, in order"
        values = [ln.split(": ", 1)[1] for ln in lines]
        assert all(values) and "nan" not in values
        gold = ds.classes[ds.y[ex_id]]
        assert gold not in values and gold not in text.split(), "the gold label never appears"


def test_german_decoded_values():
    cfg = config.load("datasets")["german"]
    ds = datasets.load("german")
    assert ds.features == [cfg["feature_names"][c] for c in cfg["feature_names"]], "every OpenML column has a display name"
    for ex in ds.jev_ids:
        values = dict(ln.split(": ", 1) for ln in serialize(ds, ex).splitlines())
        for col, mapping in cfg["value_labels"].items():
            codes = {raw for raw, label in mapping.items() if raw != label}
            assert values[cfg["feature_names"][col]] not in codes, f"raw OpenML code in {col}"


def test_missing_value_is_written_as_unknown():
    ds = datasets.build("wdbc")
    import copy
    import numpy as np
    clone = copy.copy(ds)
    clone.frame = ds.frame.copy()
    ex = ds.jev_ids[0]
    clone.frame.loc[ex, "mean radius"] = np.nan
    assert f"mean radius: {MISSING_VALUE}" in serialize(clone, ex)


def test_masking_removes_features():
    ds = datasets.load("wdbc")
    ex = ds.jev_ids[0]
    masked = serialize(ds, ex, drop={0, 2})
    assert len(masked.splitlines()) == len(ds.features) - 2
    assert not masked.startswith(ds.features[0] + ":") and ds.features[1] + ":" in masked


@pytest.mark.parametrize("name", ["sst2", "agnews"])
def test_text_serialization_and_masking(name):
    ds = datasets.load(name)
    for ex in ds.eval_ids[:200]:
        text = serialize(ds, ex)
        assert text == ds.frame.at[ex, "text"] and text == " ".join(text.split()) and text, "normalised, non-empty"
    ex = ds.eval_ids[0]
    words = units(ds, ex)
    assert serialize(ds, ex, drop={0}).split() == words[1:]
    assert serialize(ds, ex, drop=range(len(words))) == ""
