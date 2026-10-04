"""Removing a dataset after work was paid for on it: its share is recorded first, the other
datasets' frozen identifiers do not change, and every total keeps it."""
import argparse
import json

import pytest

from ciayn import config, remove_dataset
from ciayn.data import manifest
from ciayn.paper import provenance


H_KEPT, H_GONE = manifest.sha256(b'{"a":1}'), manifest.sha256(b'{"b":2}')


def fake_spending(workspace):
    log = config.CACHE / "exchanges"
    log.mkdir(parents=True)
    rows = [{"stage": "predict", "dataset": "gone", "status": 200, "response": {"usage": {"input_tokens": 1000}}},
            {"stage": "predict", "dataset": "gone", "status": 500, "response": None},
            {"stage": "explain", "dataset": "gone", "status": 200, "response": {"usage": {"input_tokens": 3000}}},
            {"stage": "predict", "dataset": "kept", "status": 200, "response": {"usage": {"input_tokens": 9999}}}]
    (log / "predict.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    attempts = [{"dataset": "gone", "status": "Succeeded", "cost_usd": 0.05},
                {"model": "bert", "dataset": "gone", "status": "Succeeded", "cost_usd": 0.15},
                {"model": "qwen", "kind": "smoke", "dataset": "kept,gone", "status": "Stopped", "cost_usd": 3.8},
                {"model": "bert", "dataset": "kept", "status": "Succeeded", "cost_usd": 1.0}]
    (config.results_dir("live") / "baseline_attempts.json").write_text(json.dumps(attempts))


def test_removed_share_counts_only_that_dataset(workspace):
    fake_spending(workspace)
    r = provenance.write_removed("gone")
    price = config.load("jev")["usd_per_mtok_input"]
    assert r["jev"]["requests"] == 3 and r["jev"]["by_stage"]["predict"] == {"requests": 2, "failed": 1, "input_tokens": 1000,
                                                                            "usd": 1000 * price / 1e6}
    assert r["jev"]["input_tokens"] == 4000 and r["jev"]["usd"] == pytest.approx(4000 * price / 1e6)
    assert r["aws"]["dedicated_usd"] == pytest.approx(0.20) and [a["model"] for a in r["aws"]["dedicated_executions"]] == ["xgb", "bert"]
    assert r["aws"]["shared_usd"] == pytest.approx(3.8), "a job that also covered other datasets is listed whole, as shared"
    saved = json.loads(provenance.removed_path().read_text())
    assert saved["datasets"]["gone"]["jev"]["requests"] == 3


def frozen(workspace, monkeypatch):
    splits = workspace / "splits"
    splits.mkdir()
    monkeypatch.setattr(config, "SPLITS", splits)
    (splits / "kept.json").write_text('{"a":1}')
    (splits / "gone.json").write_text('{"b":2}')
    text = "# frozen 2026-09-30T13:20:53Z\n" + f"{H_GONE}  gone.json\n" + f"{H_KEPT}  kept.json\n"
    (splits / manifest.MANIFEST).write_text(text)
    return splits, text


def test_removal_is_refused_until_the_spending_is_recorded(workspace, monkeypatch):
    splits, text = frozen(workspace, monkeypatch)
    with pytest.raises(manifest.ManifestError, match="has not been recorded"):
        manifest.remove_dataset("gone")
    assert (splits / manifest.MANIFEST).read_text() == text and (splits / "gone.json").exists()


def test_removal_leaves_the_other_datasets_frozen_files_untouched(workspace, monkeypatch):
    splits, text = frozen(workspace, monkeypatch)
    fake_spending(workspace)
    provenance.write_removed("gone")
    for mode in ("live", "mock"):
        (config.results_dir(mode, "gone", "jev") / "predictions.json").write_text("{}")
    monkeypatch.setattr(config, "DATASETS", ("kept",))
    remove_dataset.run(argparse.Namespace(removed="gone"))
    assert (splits / manifest.MANIFEST).read_text() == text.replace(f"{H_GONE}  gone.json\n", "")
    assert manifest.frozen_at() == "2026-09-30T13:20:53Z", "the freeze time is not rewritten"
    assert (splits / "kept.json").read_text() == '{"a":1}' and not (splits / "gone.json").exists()
    rec = json.loads((splits / "REMOVED.json").read_text())["removed"][0]
    assert rec["dataset"] == "gone" and rec["unchanged_sha256"] == {"kept.json": H_KEPT}
    assert not (config.RESULTS / "live" / "gone").exists() and not (config.RESULTS / "mock" / "gone").exists()
    assert provenance.removed_path().exists(), "the record of what was spent survives the cleanup"


def test_a_dataset_still_configured_is_not_removed(workspace, monkeypatch):
    monkeypatch.setattr(config, "DATASETS", ("gone",))
    with pytest.raises(RuntimeError, match="still in config.DATASETS"):
        remove_dataset.run(argparse.Namespace(removed="gone"))
