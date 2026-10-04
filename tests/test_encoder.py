"""BERT baseline: packaging, the job script on a tiny model on CPU, pipeline and run logic. No AWS, no download."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ciayn import config
from ciayn.baseline import package as xgb_package
from ciayn.baseline.scripts.train import nested_subset
from ciayn.encoder import package


def test_packaged_test_examples_carry_no_labels_and_use_jevs_text(tmp_path, german):
    from ciayn.data.serialize import serialize
    root = package.write_inputs(german, tmp_path)
    job = json.loads((root / "job.json").read_text())
    assert job["max_length"] == 512 and job["revision"] == config.load("encoder")["revision"]
    for r in (0, 9):
        rows = [json.loads(x) for x in (root / f"repeat_{r}" / "test.jsonl").read_text().splitlines()]
        assert rows and all(set(x) == {"id", "text"} for x in rows), "no label in the evaluation rows"
        assert [x["id"] for x in rows] == german.repeats[r]["test"]
        assert rows[0]["text"] == serialize(german, rows[0]["id"]), "BERT reads the text Jev receives"
        train = [json.loads(x) for x in (root / f"repeat_{r}" / "train.jsonl").read_text().splitlines()]
        assert all("y" in x for x in train)


def test_training_fractions_are_the_xgboost_ones(tmp_path, german):
    xgb_root = xgb_package.write_inputs(german, tmp_path / "xgb")
    seed = config.load("baseline")["seed"]
    for r in (0, 4):
        y_xgb = pd.read_csv(xgb_root / f"repeat_{r}" / "train.csv")["y"].to_numpy()
        ours = package.fraction_indices(german, r)
        for f in config.load("baseline")["fractions"]:
            assert ours[str(f)] == nested_subset(y_xgb, f, seed + r).tolist(), (r, f)


def tiny_bert(path: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    words = "the a good bad movie loan credit checking account months duration yes no unknown".split()
    (path / "vocab.txt").parent.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", "."] + words) + "\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    cfg = BertConfig(vocab_size=7 + len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    BertForSequenceClassification(cfg).save_pretrained(path)
    return path


def toy_job(root: Path, n_repeats=2) -> Path:
    rng = np.random.default_rng(0)
    words = ["good movie", "bad movie", "good loan credit", "bad loan account"]
    job = {"dataset": "toy", "n_classes": 2, "n_repeats": n_repeats, "model_id": "tiny", "revision": None,
           "grid": {"learning_rate": [1e-3, 5e-3]}, "max_epochs": 2, "batch_size": 8, "eval_batch_size": 16,
           "warmup_ratio": 0.1, "weight_decay": 0.0, "seed": 0, "max_length": 8, "fractions": [0.5, 1.0],
           "time_budget_s": 3600}
    root.mkdir(parents=True)
    (root / "job.json").write_text(json.dumps(job))
    for r in range(n_repeats):
        d = root / f"repeat_{r}"
        d.mkdir()
        for split, n in (("train", 24), ("val", 8), ("test", 8)):
            with open(d / f"{split}.jsonl", "w") as f:
                for i in range(n):
                    c = int(rng.integers(0, 4))
                    rec = {"id": f"{split}{r}-{i}", "text": words[c] + " yes no unknown the a months duration"}
                    if split != "test":
                        rec["y"] = c % 2
                    f.write(json.dumps(rec) + "\n")
        (d / "fractions.json").write_text(json.dumps({"0.5": list(range(0, 24, 2)), "1.0": list(range(24))}))
    return root


def test_bert_job_script_runs_on_a_tiny_model(tmp_path, monkeypatch):
    from ciayn.encoder.scripts import bert_train
    monkeypatch.setenv("BERT_MODEL_DIR", str(tiny_bert(tmp_path / "model")))
    src, out = toy_job(tmp_path / "in"), tmp_path / "out"
    bert_train.main(src, out)
    preds, curve = pd.read_csv(out / "predictions.csv"), pd.read_csv(out / "learning_curve.csv")
    run = json.loads((out / "run.json").read_text())
    assert np.allclose(preds[["p0", "p1"]].sum(axis=1), 1, atol=1e-6)
    assert set(preds.groupby(["repeat", "split"]).size()) == {8} and len(preds) == 2 * 2 * 8
    sel = run["selected"]
    assert sel["learning_rate"] in (1e-3, 5e-3) and 1 <= sel["epochs"] <= 2
    assert sel["val_logloss"] == pytest.approx(min(g["val_logloss"] for g in run["repeats"]["0"]["grid"]))
    assert run["repeats"]["0"]["label_budget"]["early_stopping_and_selection"] == 8
    assert run["repeats"]["1"]["label_budget"]["early_stopping_and_selection"] == 0, "repeat 1 reuses repeat 0's choice"
    assert sorted(curve.n_train.unique()) == [12, 24] and run["repeats"]["0"]["infer_s_per_item"] > 0
    full = curve[(curve.repeat == 0) & (curve.fraction == 1.0)].set_index("id")[["p0", "p1"]]
    test = preds[(preds.repeat == 0) & (preds.split == "test")].set_index("id")[["p0", "p1"]]
    assert np.allclose(full.loc[test.index], test), "the full-fraction curve point is the reported model"


def test_bert_job_refuses_labelled_test_rows(tmp_path, monkeypatch):
    from ciayn.encoder.scripts import bert_train
    monkeypatch.setenv("BERT_MODEL_DIR", str(tiny_bert(tmp_path / "model")))
    src = toy_job(tmp_path / "in", n_repeats=1)
    rows = [json.loads(x) for x in (src / "repeat_0" / "test.jsonl").read_text().splitlines()]
    (src / "repeat_0" / "test.jsonl").write_text("\n".join(json.dumps({**x, "y": 0}) for x in rows))
    with pytest.raises(RuntimeError, match="must not carry labels"):
        bert_train.main(src, tmp_path / "out")


def test_pipeline_definition_is_deterministic_and_pinned():
    from ciayn.encoder import pipeline
    a = pipeline.definition("us-east-2", "bucket", "arn:role")
    assert a == pipeline.definition("us-east-2", "bucket", "arn:role")
    spec = a["Steps"][0]["Arguments"]["AlgorithmSpecification"]
    cfg = config.load("encoder")
    assert spec["TrainingImage"] == f"763104351884.dkr.ecr.us-east-2.amazonaws.com/{cfg['image']}"
    assert [s["Type"] for s in a["Steps"]] == ["Training"]
    assert pipeline.worst_case_cost("ml.g5.xlarge", "us-east-2") == pytest.approx(3 * 1.2575)


def _bert_outputs(ds, k=2):
    rows = [{"id": i, "repeat": r, "split": s, **{f"p{c}": 1 / k for c in range(k)}}
            for r, rep in enumerate(ds.repeats) for s in ("val", "test") for i in rep[s]]
    curve = [{"id": i, "repeat": r, "fraction": 1.0, "n_train": len(rep["train"]), **{f"p{c}": 1 / k for c in range(k)}}
             for r, rep in enumerate(ds.repeats) for i in rep["test"]]
    run = {"repeats": {str(r): {"label_budget": {"training": len(rep["train"]), "early_stopping_and_selection": 0,
                                                 "calibration": 0, "threshold_selection": 0}, "infer_s_per_item": 0.001}
                       for r, rep in enumerate(ds.repeats)}, "selected": {"learning_rate": 2e-5, "epochs": 2}}
    return {"predictions.csv": pd.DataFrame(rows).to_csv(index=False).encode(),
            "learning_curve.csv": pd.DataFrame(curve).to_csv(index=False).encode(), "run.json": json.dumps(run).encode()}


def test_run_iterates_combinations_then_fetch_checks_identifiers(approved, monkeypatch, wdbc):
    from conftest import stage_args
    from fakes import FakeAws
    from ciayn.baseline import cli as aws_cli
    from ciayn.encoder import cli
    LIMIT = "ResourceLimitExceeded: The account-level service limit 'ml.g5.xlarge for training job usage' is 0 Instances"
    FakeAws.outcomes[("us-east-2", "ml.g5.xlarge")] = LIMIT
    files = _bert_outputs(wdbc)
    monkeypatch.setattr(FakeAws, "download", lambda self, b, k, path: _tar(path, files), raising=False)
    assert cli.run_live(stage_args("live", "wdbc"), make=FakeAws) == 0
    recs = aws_cli.attempts("bert")
    assert [(a["instance_type"], a["status"]) for a in recs] == [("ml.g5.xlarge", "Failed"), ("ml.g5.2xlarge", "Succeeded")]
    assert json.loads(cli.combination_path().read_text())["instance_type"] == "ml.g5.2xlarge"
    assert not aws_cli.running_marker("wdbc", "bert").exists()
    cli.fetch(stage_args("live", "wdbc"), make=FakeAws)
    from ciayn import results
    df = results.read_predictions("live", "wdbc", "bert")
    assert len(df) == sum(len(r["val"]) + len(r["test"]) for r in wdbc.repeats)
    bad = dict(files)
    frame = pd.read_csv(__import__("io").BytesIO(files["predictions.csv"]))
    bad["predictions.csv"] = frame.iloc[1:].to_csv(index=False).encode()
    with pytest.raises(RuntimeError, match="do not match the frozen identifiers"):
        cli.ingest(wdbc, bad, {})


def test_dry_run_and_spending_limit_start_nothing(approved, monkeypatch, capsys):
    from conftest import stage_args
    from fakes import FakeAws
    from ciayn.encoder import cli
    assert cli.run_live(stage_args("live", "german", dry_run=True), make=FakeAws) == 0
    assert "would start 1 training job" in capsys.readouterr().out and FakeAws.log == []
    monkeypatch.setitem(config.load("budgets"), "aws_spend_limit_usd", 1.0)
    with pytest.raises(RuntimeError, match="spending limit"):
        cli.run_live(stage_args("live", "german"), make=FakeAws)
    assert FakeAws.log == []


def _tar(path, files):
    import io
    import tarfile
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
