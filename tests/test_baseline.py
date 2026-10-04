"""Supervised baseline: job scripts on toy data, packaging, pipeline definition and the run logic. No AWS."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fakes import FakeAws

from ciayn import backup, config, results
from ciayn.baseline import cli, package, pipeline
from ciayn.baseline.scripts import preprocess, train
from ciayn.data import datasets
from conftest import stage_args


# ---- processing script ---------------------------------------------------------------------------

def _write(d: Path, **frames):
    d.mkdir(parents=True)
    for name, f in frames.items():
        f.to_csv(d / f"{name}.csv", index=False)


def test_tabular_preprocessing_is_fitted_on_training_rows_only(tmp_path):
    tr = pd.DataFrame({"id": ["a", "b"], "y": [0, 1], "colour": ["red", "blue"], "size": [1.0, 2.0]})
    va = pd.DataFrame({"id": ["c"], "y": [0], "colour": ["green"], "size": [3.0]})          # unseen category
    te = pd.DataFrame({"id": ["d"], "colour": ["none"], "size": [4.0]})                      # "none" is a value, not missing
    _write(tmp_path / "in" / "repeat_0", train=tr, val=va, test=te)
    job = {"dataset": "toy", "kind": "tabular", "n_repeats": 1, "numeric": ["size"], "categorical": ["colour"]}
    transform, names = preprocess.fit_tabular(tr, job)
    assert names == ["size", "colour=blue", "colour=red"], "a category seen only outside training has no fitted entry"
    info = preprocess.process_repeat(tmp_path / "in" / "repeat_0", tmp_path / "out" / "repeat_0", job)
    Xv = sp.load_npz(tmp_path / "out" / "repeat_0" / "X_val.npz").toarray()
    assert Xv.tolist() == [[3.0, 0.0, 0.0]] and info == {"n_features": 3, "n_train": 2, "n_val": 1, "n_test": 1}
    assert not (tmp_path / "out" / "repeat_0" / "y_test.npy").exists()


def test_text_vocabulary_comes_from_training_texts_only():
    tr = pd.DataFrame({"id": ["a", "b"], "y": [0, 1], "text": ["good film", "bad film"]})
    te = pd.DataFrame({"id": ["c"], "text": ["zebra film"]})
    transform, names = preprocess.fit_text(tr, {"ngram_range": [1, 2], "max_features": 100})
    assert "zebra" not in names and "film" in names and "good film" in names
    X = transform(te)
    assert X.shape == (1, len(names)) and X.nnz == 1, "only the word known from training is represented"


def test_processing_refuses_test_rows_with_labels(tmp_path):
    tr = pd.DataFrame({"id": ["a", "b"], "y": [0, 1], "text": ["x y", "z w"]})
    _write(tmp_path / "repeat_0", train=tr, val=tr, test=tr)
    with pytest.raises(ValueError, match="must not contain labels"):
        preprocess.process_repeat(tmp_path / "repeat_0", tmp_path / "out", {"kind": "text", "ngram_range": [1, 1], "max_features": 10})


# ---- packaging -----------------------------------------------------------------------------------

def test_package_withholds_test_labels(tmp_path, wdbc):
    root = package.write_inputs(wdbc, tmp_path)
    job = json.loads((root / "job.json").read_text())
    assert job["n_repeats"] == 10 and job["kind"] == "tabular" and len(job["numeric"]) == 30 and job["categorical"] == []
    for r in (0, 9):
        tr, va, te = (pd.read_csv(root / f"repeat_{r}" / f"{s}.csv") for s in ("train", "val", "test"))
        assert "y" in tr and "y" in va and "y" not in te
        assert te["id"].tolist() == wdbc.ids("test", r) and len(tr) == 341 and len(te) == 114


def test_heldout_evaluation_labels_are_not_in_the_package(tmp_path):
    ds = datasets.load("sst2")
    root = package.write_inputs(ds, tmp_path)
    held = set(ds.eval_ids)
    labelled = set()
    for f in root.rglob("*.csv"):
        df = pd.read_csv(f, usecols=lambda c: c in ("id", "y"))
        if "y" in df:
            labelled |= set(df["id"])
    assert not labelled & held, "no file in the package pairs an evaluation example with its label"
    assert json.loads((root / "job.json").read_text())["n_classes"] == 2


# ---- training script, end to end on a real small dataset ------------------------------------------

@pytest.fixture
def quick(monkeypatch):
    cfg = config.load("baseline")
    monkeypatch.setitem(cfg, "grid", {"max_depth": [3], "eta": [0.3]})
    monkeypatch.setitem(cfg, "max_rounds", 40)
    monkeypatch.setitem(cfg, "patience", 5)
    monkeypatch.setitem(cfg, "fractions", [0.25, 1.0])


def test_scripts_end_to_end_on_breast_cancer(tmp_path, workspace, wdbc, quick):
    root = package.write_inputs(wdbc, tmp_path / "input")
    preprocess.main(root, tmp_path / "features")
    rec = train.main(tmp_path / "features", tmp_path / "out")
    assert set(rec["repeats"]) == {str(r) for r in range(10)}
    lb = rec["repeats"]["3"]["label_budget"]
    assert lb == {"training": 341, "early_stopping_and_selection": 114, "threshold_selection": 114, "calibration": 0}
    assert rec["repeats"]["0"]["curve_train_sizes"] == {"0.25": 86, "1.0": 341}
    assert rec["repeats"]["0"]["infer_s_per_item"] > 0, "batch inference is timed per evaluation item"
    cli.ingest(wdbc, tmp_path / "out" / "predictions.csv", tmp_path / "out" / "learning_curve.csv",
               {**rec, "model_version": "xgboost-test"}, mode="mock")
    for r in range(10):
        s = results.scores("mock", wdbc, "xgb", "test", r)
        assert s.ids == wdbc.ids("test", r) and np.allclose(s.P.sum(1), 1, atol=1e-4)
        assert np.mean(s.pred == s.y) > 0.88
    curve = pd.read_parquet(results.model_dir("mock", "wdbc", "xgb") / "learning_curve.parquet")
    assert sorted(curve.fraction.unique()) == [0.25, 1.0] and len(curve) == 10 * 2 * 114
    run = json.loads((results.model_dir("mock", "wdbc", "xgb") / "run.json").read_text())
    assert run["repeats"]["0"]["selected"]["max_depth"] == 3


def test_fetch_refuses_predictions_with_other_identifiers(tmp_path, workspace, wdbc, quick):
    root = package.write_inputs(wdbc, tmp_path / "input")
    preprocess.main(root, tmp_path / "features")
    train.main(tmp_path / "features", tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    df.iloc[1:].to_csv(tmp_path / "out" / "short.csv", index=False)
    with pytest.raises(RuntimeError, match="do not match the frozen identifiers"):
        cli.ingest(wdbc, tmp_path / "out" / "short.csv", tmp_path / "out" / "learning_curve.csv", {}, mode="mock")


def test_nested_subsets_are_nested_and_stratified():
    y = np.array([0] * 80 + [1] * 20)
    small, large = train.nested_subset(y, 0.1, 0), train.nested_subset(y, 0.5, 0)
    assert set(small) <= set(large) and len(small) == 10 and y[small].sum() == 2


# ---- pipeline definition and the run logic -------------------------------------------------------

def test_pipeline_definition_is_reproducible_and_has_two_jobs():
    a = pipeline.definition("us-east-2", "bucket-x", "arn:aws:iam::1:role/r")
    b = pipeline.definition("us-east-2", "bucket-x", "arn:aws:iam::1:role/r")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert [(s["Name"], s["Type"]) for s in a["Steps"]] == [("Preprocess", "Processing"), ("Train", "Training")]
    assert a["Steps"][0]["Arguments"]["AppSpecification"]["ImageUri"].startswith("257758044811.dkr.ecr.us-east-2")
    assert "sagemaker-xgboost:3.0-5" in a["Steps"][1]["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert pipeline.definition("eu-west-1", "b", "r")["Steps"][1]["Arguments"]["AlgorithmSpecification"]["TrainingImage"].startswith("141502667606")
    assert {p["Name"] for p in a["Parameters"]} == {"Dataset", "RunId", "InstanceType"}


LIMIT = "ClientError: ResourceLimitExceeded: The account-level service limit 'ml.m5.xlarge for processing job usage' is 0 Instances"


def test_refused_instance_types_and_regions_are_skipped_until_one_works(approved):
    cfg = config.load("baseline")
    for it in cfg["instance_types"]:
        FakeAws.outcomes[("us-east-2", it)] = LIMIT
    FakeAws.outcomes[("us-east-1", "ml.m5.xlarge")] = "CapacityError: Unable to provision requested ML compute capacity"
    FakeAws.quotas[("us-east-1", "ml.m5.large")] = 0.0                       # readable quota of zero: skipped, not started
    assert cli.run_live(stage_args("live", "german"), make=FakeAws) == 0
    combo = json.loads((config.results_dir("live") / "baseline_combination.json").read_text())
    assert (combo["region"], combo["instance_type"]) == ("us-east-1", "ml.m5.2xlarge")
    log = cli.attempts()
    assert len(log) == 6 + 1 + 1 + 1 and [a["status"] for a in log][-3:] == ["Failed", "Skipped", "Succeeded"]
    assert all(a["refused"] for a in log[:-1]) and log[-1]["billable_seconds"] == 180.0
    assert ("delete_stack", "us-east-2") in FakeAws.log, "the stack in the region that was not used is removed"
    assert cli.stack_state()["region"] == "us-east-1" and list(cli.stack_state()["stacks"]) == ["us-east-1"]
    # the other datasets then run on the same combination, without probing again
    FakeAws.log.clear()
    cli.run_live(stage_args("live", "wdbc"), make=FakeAws)
    assert FakeAws.log == [("start", "us-east-1", "ml.m5.2xlarge", "wdbc")]


def test_all_combinations_refused_stops_with_every_attempt(approved):
    cfg = config.load("baseline")
    FakeAws.outcomes = {(r, it): LIMIT for r in cfg["regions"] for it in cfg["instance_types"]}
    with pytest.raises(RuntimeError, match="refused every configured combination"):
        cli.run_live(stage_args("live", "german"), make=FakeAws)
    assert len(cli.attempts()) == 24 and not (config.results_dir("live") / "baseline_combination.json").exists()


def test_a_failure_in_our_own_job_does_not_trigger_the_iteration(approved):
    FakeAws.outcomes[("us-east-2", "ml.m5.xlarge")] = "AlgorithmError: train.py exited with code 1"
    with pytest.raises(RuntimeError, match="not instance or region availability"):
        cli.run_live(stage_args("live", "german"), make=FakeAws)
    assert len(cli.attempts()) == 1


def test_spending_limit_blocks_before_any_job(approved, monkeypatch):
    monkeypatch.setitem(config.load("budgets"), "aws_spend_limit_usd", 0.10)
    with pytest.raises(RuntimeError, match="AWS spending limit"):
        cli.run_live(stage_args("live", "german"), make=FakeAws)
    assert not [e for e in FakeAws.log if e[0] == "start"]


def test_dry_run_lists_jobs_and_starts_nothing(approved, capsys):
    args = stage_args("live", "all")
    args.dry_run = True
    assert cli.run_live(args, make=FakeAws) == 0
    out = capsys.readouterr().out
    assert out.count("would start 1 processing job and 1 training job") == len(config.DATASETS) and FakeAws.log == []


def test_deploy_and_teardown_need_approval_and_teardown_needs_a_matching_backup(approved):
    with pytest.raises(RuntimeError, match="only after they approve"):
        cli.deploy(stage_args("live", yes=False))
    with pytest.raises(RuntimeError, match="once the user confirms"):
        cli.teardown(stage_args("live", yes=False))
    with pytest.raises(RuntimeError, match="teardown refused: no backup has been recorded"):
        cli.teardown(stage_args("live", yes=True))
    (config.CACHE / "exchanges").mkdir(parents=True)
    (config.CACHE / "exchanges" / "predict.jsonl").write_text("{}\n")
    (config.results_dir("live") / "aws_stack.json").write_text("{}")            # no bucket yet: local backup
    backup.run()
    (config.CACHE / "exchanges" / "predict.jsonl").write_text("{}\n{}\n")
    with pytest.raises(RuntimeError, match="changed since the latest backup"):
        cli.teardown(stage_args("live", yes=True))


def test_run_without_an_approved_template_is_refused(workspace):
    FakeAws.log = []
    with pytest.raises(RuntimeError, match="has not been approved"):
        cli.run_live(stage_args("live", "german"), make=FakeAws)
    assert FakeAws.log == []


def test_text_jobs_use_coarse_histograms_and_tabular_jobs_the_default(wdbc, sst2):
    assert package.job_spec(sst2)["max_bin"] == 16 and "max_bin" not in package.job_spec(wdbc)
    assert package.job_spec(sst2)["time_budget_s"] == pytest.approx(0.8 * 10800)
    assert train.params(2, 3, 0.1, 0, 16)["max_bin"] == 16 and train.params(4, 3, 0.1, 0)["max_bin"] == 256


def test_learning_curve_fits_are_skipped_and_recorded_when_the_time_budget_is_used_up(tmp_path, workspace, wdbc, quick, monkeypatch):
    monkeypatch.setitem(config.load("baseline"), "curve_time_share", 0.0)     # no time left for curve fits
    root = package.write_inputs(wdbc, tmp_path / "input")
    preprocess.main(root, tmp_path / "features")
    rec = train.main(tmp_path / "features", tmp_path / "out")
    r0 = rec["repeats"]["0"]
    assert r0["curve_train_sizes"] == {"1.0": 341} and "0.25" in r0["curve_skipped"]
    assert "time budget" in r0["curve_skipped"]["0.25"]
    assert len(pd.read_csv(tmp_path / "out" / "predictions.csv")) == 10 * (114 + 114), "predictions are complete regardless"
