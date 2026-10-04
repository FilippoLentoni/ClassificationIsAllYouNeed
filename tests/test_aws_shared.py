"""One record of SageMaker attempts and one spending limit for XGBoost, BERT and Qwen
(supervised-baseline: AWS resource lifecycle and cost)."""
import json

import pytest
from fakes import FakeAws

from ciayn import config
from ciayn.baseline import cli


def write_attempts(recs):
    (config.results_dir("live") / "baseline_attempts.json").write_text(json.dumps(recs))


def test_records_without_a_model_are_xgboost_and_the_total_covers_every_model(approved):
    write_attempts([{"dataset": "german", "status": "Succeeded", "cost_usd": 0.25},
                    {"model": "bert", "dataset": "german", "status": "Succeeded", "cost_usd": 1.5},
                    {"model": "qwen", "dataset": "all", "status": "Stopped", "cost_usd": 10.0}])
    assert [a["model"] for a in cli.attempts()] == ["xgb", "bert", "qwen"]
    assert cli.recorded_cost() == pytest.approx(11.75), "stopped executions are paid for and counted"
    assert len(cli.attempts("bert")) == 1 and cli.succeeded("german", "bert")["model"] == "bert"
    assert cli.succeeded("german")["model"] == "xgb"
    with pytest.raises(ValueError, match="must name its model"):
        cli.record_attempt({"dataset": "german"})


def test_a_job_whose_worst_case_exceeds_the_remaining_limit_is_refused(approved, monkeypatch):
    monkeypatch.setitem(config.load("budgets"), "aws_spend_limit_usd", 75.0)
    write_attempts([{"model": "xgb", "cost_usd": 0.42}, {"model": "bert", "cost_usd": 20.0}])
    cli.check_spend("ml.g6e.12xlarge", "us-east-2", worst_case_usd=54.0)        # 74.42 <= 75
    with pytest.raises(RuntimeError, match="exceeds the limit of \\$75.00"):
        cli.check_spend("ml.g6e.12xlarge", "us-east-2", worst_case_usd=55.0)


def test_run_pipeline_prices_each_step_on_its_own_instance(approved):
    aws = FakeAws("us-east-2")
    stack = {"bucket": "b", "role_arn": "arn:role"}
    rec = cli.run_pipeline("qwen", "all", "us-east-2", "ml.g6e.12xlarge", aws, stack, "ciayn-qwen", {}, {"Dataset": "all"},
                           instance_types={"Preprocess": "ml.m5.xlarge"})
    prices = config.load("baseline")["price_usd_per_hour"]
    assert rec["model"] == "qwen" and rec["status"] == "Succeeded"
    assert rec["cost_usd"] == pytest.approx(60 / 3600 * prices["ml.m5.xlarge"] + 120 / 3600 * prices["ml.g6e.12xlarge"])
    assert cli.attempts("qwen")[-1]["execution_arn"] == rec["execution_arn"]


def test_operator_policy_names_exactly_the_three_pipelines():
    import re
    text = (config.ROOT / "infra" / "sagemaker.yaml").read_text()
    names = set(re.findall(r":pipeline/([a-z0-9-]+)['/]", text))
    assert names == {"ciayn-xgb", "ciayn-bert", "ciayn-qwen"} and "pipeline/ciayn-*" not in text


def test_runs_refuse_a_template_that_differs_from_the_deployed_one(approved, monkeypatch):
    state = {"approved_at": "2026-09-30T00:00:00Z",
             "stacks": {"us-east-2": {"stack": "s", "bucket": "b", "role_arn": "r", "template_sha256": "old"}}}
    (config.results_dir("live") / "aws_stack.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="differs from the template deployed"):
        cli.ensure_stack("us-east-2", FakeAws("us-east-2"))
    monkeypatch.setattr(FakeAws, "update_stack", lambda self, *a: {"BucketName": "b", "ExecutionRoleArn": "r"}, raising=False)
    monkeypatch.setattr(cli, "make_aws", FakeAws)
    import argparse
    with pytest.raises(RuntimeError, match="only after they approve"):
        cli.update(argparse.Namespace(yes=False))
    cli.update(argparse.Namespace(yes=True))
    assert cli.ensure_stack("us-east-2", FakeAws("us-east-2"))["template_sha256"] == cli.template_sha()


def test_teardown_removes_all_three_pipelines_and_the_bucket(approved, monkeypatch):
    import argparse
    from ciayn import backup
    state = {"approved_at": "x", "stacks": {"us-east-2": {"stack": "ciayn-baseline", "bucket": "b", "role_arn": "r"}}}
    (config.results_dir("live") / "aws_stack.json").write_text(json.dumps(state))
    monkeypatch.setattr(backup, "local_matches_latest", lambda: (True, ""))
    deleted = []
    monkeypatch.setattr(FakeAws, "delete_pipeline", lambda self, name: deleted.append(name))
    monkeypatch.setattr(cli, "make_aws", FakeAws)
    cli.teardown(argparse.Namespace(yes=True))
    assert deleted == ["ciayn-xgb", "ciayn-bert", "ciayn-qwen"]
    assert ("delete_stack", "us-east-2") in FakeAws.log and cli.stack_state()["stacks"] == {}


def test_an_execution_that_waits_too_long_for_capacity_is_stopped_and_reported(monkeypatch):
    from ciayn.baseline import aws as aws_mod

    class Sm:
        stopped = False

        def describe_pipeline_execution(self, PipelineExecutionArn):
            return {"PipelineExecutionStatus": "Stopped" if self.stopped else "Executing"}

        def list_pipeline_execution_steps(self, PipelineExecutionArn):
            return {"PipelineExecutionSteps": [{"StepName": "Infer", "StepStatus": "Stopped" if self.stopped else "Executing",
                                                "Metadata": {"TrainingJob": {"Arn": "arn:x/job-1"}}}]}

        def describe_training_job(self, TrainingJobName):
            return {"SecondaryStatus": "Pending", "SecondaryStatusTransitions": [{"StatusMessage": "Training job waiting for capacity"}],
                    "AlgorithmSpecification": {"TrainingImage": "img"}, "BillableTimeInSeconds": 0,
                    "OutputDataConfig": {"S3OutputPath": "s3://b/o"}}

        def stop_pipeline_execution(self, PipelineExecutionArn):
            self.stopped = True

    a = aws_mod.Aws.__new__(aws_mod.Aws)
    a.sm = Sm()
    clock = iter(range(0, 10_000, 600))
    monkeypatch.setattr(aws_mod.time, "time", lambda: next(clock))
    monkeypatch.setattr(aws_mod.time, "sleep", lambda s: None)
    out = a.wait("arn", capacity_timeout_s=1800)
    assert a.sm.stopped and out["status"] == "Stopped" and out["failure"].startswith("CapacityError")
    assert cli.refused(out["failure"]), "a capacity timeout moves the iteration to the next instance type"
    assert out["steps"]["Infer"]["billable_seconds"] == 0.0
