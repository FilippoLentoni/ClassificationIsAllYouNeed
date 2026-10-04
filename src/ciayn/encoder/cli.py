"""`encoder plan | run | fetch` (design D2): BERT fine-tuned on SageMaker.

`run` tries the configured GPU instance types, then regions, in order and without asking, on the
first dataset that has not succeeded yet, records every attempt, and runs the remaining datasets
on the combination that worked. The spending limit covers every SageMaker job of the study.
"""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from .. import config, templates
from ..baseline import cli as aws_cli
from ..data import datasets
from ..data.datasets import Dataset
from ..results import model_dir, prob_columns, write_supervised_predictions
from . import package, pipeline

MODEL = "bert"
SCRIPTS = Path(__file__).parent / "scripts"


def combination_path() -> Path:
    return aws_cli._live("encoder_combination.json")


def plan(args) -> int:
    cfg = config.load("encoder")
    region, itype = cfg["regions"][0], cfg["instance_types"][0]
    for n in templates.selected(args):
        print(f"encoder plan: {n}: 1 training job ({cfg['model_id']} at {cfg['revision'][:12]}), worst case "
              f"${pipeline.worst_case_cost(itype, region):.2f} on {itype}")
    print(f"encoder plan: tried in order, instance types {cfg['instance_types']} in regions {cfg['regions']}; "
          f"spending limit ${config.load('budgets')['aws_spend_limit_usd']:.2f}, recorded so far ${aws_cli.recorded_cost():.2f}")
    return 0


def execute(ds: Dataset, region: str, itype: str, aws, stack: dict) -> dict:
    aws_cli.check_spend(itype, region, pipeline.worst_case_cost(itype, region))
    marker = aws_cli.running_marker(ds.name, MODEL)
    marker.write_text(json.dumps({"dataset": ds.name, "started": aws_cli.now()}))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            aws.upload_dir(package.write_inputs(ds, Path(tmp)), stack["bucket"], f"input-bert/{ds.name}")
        aws.upload_dir(SCRIPTS, stack["bucket"], "code-bert")
        return aws_cli.run_pipeline(MODEL, ds.name, region, itype, aws, stack, config.load("encoder")["pipeline_name"],
                                    pipeline.definition(region, stack["bucket"], stack["role_arn"]), {"Dataset": ds.name})
    finally:
        marker.unlink(missing_ok=True)


def run_live(args, make=aws_cli.make_aws) -> int:
    cfg = config.load("encoder")
    todo = [n for n in templates.selected(args) if args.rerun or not aws_cli.succeeded(n, MODEL)]
    if args.dry_run:
        itype = cfg["instance_types"][0]
        for n in todo:
            print(f"[dry-run] encoder run: {n}: would start 1 training job (worst case "
                  f"${pipeline.worst_case_cost(itype, cfg['regions'][0]):.2f})")
        print(f"[dry-run] encoder run: {len(todo)} executions; recorded cost ${aws_cli.recorded_cost():.2f}; nothing was started")
        return 0
    combo = json.loads(combination_path().read_text()) if combination_path().exists() else None
    for n in todo:
        if aws_cli.running_marker(n, MODEL).exists():
            print(f"encoder run: {n} is already running in another process; skipped", flush=True)
            continue
        ds = datasets.load(n)
        if combo is None:
            combo = _probe(ds, cfg, make)
            continue                                  # the probe execution was this dataset's run
        aws = make(combo["region"])
        rec = execute(ds, combo["region"], combo["instance_type"], aws, aws_cli.ensure_stack(combo["region"], aws))
        if rec["status"] != "Succeeded":
            raise RuntimeError(f"encoder run: {n} {rec['status']}: {rec['reason']}")
        print(f"encoder run: {n} succeeded in {rec['billable_seconds']:.0f} billable seconds (${rec['cost_usd']:.3f})", flush=True)
    print(f"encoder run: done; recorded AWS cost ${aws_cli.recorded_cost():.2f}")
    return 0


def _probe(ds: Dataset, cfg: dict, make) -> dict:
    for region in cfg["regions"]:
        aws = make(region)
        stack = aws_cli.ensure_stack(region, aws)
        for itype in cfg["instance_types"]:
            q = aws.quota(itype, "training")
            if q is not None and q < 1:
                aws_cli.record_attempt({"model": MODEL, "dataset": ds.name, "region": region, "instance_type": itype,
                                        "status": "Skipped", "reason": f"quota for {itype} training jobs is {q:g}",
                                        "refused": True, "cost_usd": 0.0, "started": aws_cli.now()})
                print(f"encoder run: {itype} in {region} skipped: training-job quota {q:g}", flush=True)
                continue
            rec = execute(ds, region, itype, aws, stack)
            if rec["status"] == "Succeeded":
                combo = {"region": region, "instance_type": itype, "found_at": aws_cli.now(), "probe_execution": rec["execution_arn"]}
                combination_path().write_text(json.dumps(combo, indent=1))
                print(f"encoder run: {ds.name} succeeded on {itype} in {region}", flush=True)
                return combo
            print(f"encoder run: {itype} in {region} did not complete: {rec['status']}: {rec['reason']}", flush=True)
            if not rec["refused"]:
                raise RuntimeError(f"the {ds.name} BERT execution failed for a reason that is not instance or region "
                                   f"availability, so trying other combinations would not help: {rec['reason']}")
    raise RuntimeError("SageMaker refused every configured combination for BERT; see results/live/baseline_attempts.json")


def ingest(ds: Dataset, files: dict[str, bytes], run: dict, mode: str = "live") -> None:
    cols = prob_columns(len(ds.classes))
    frame = pd.read_csv(io.BytesIO(files["predictions.csv"]))
    aws_cli.check_ids(ds, frame)
    curve = pd.read_csv(io.BytesIO(files["learning_curve.csv"]))
    for r, rep in enumerate(ds.repeats):
        if set(curve[curve.repeat == r]["id"]) - set(rep["test"]):
            raise RuntimeError(f"{ds.name}: BERT learning-curve predictions contain identifiers outside the frozen test part")
    write_supervised_predictions(mode, ds, MODEL, frame[["id", "repeat", "split"] + cols], run)
    curve.to_parquet(model_dir(mode, ds.name, MODEL) / "learning_curve.parquet", index=False)


def fetch(args, make=aws_cli.make_aws) -> int:
    cfg = config.load("encoder")
    for n in templates.selected(args):
        ds = datasets.load(n)
        if args.mode == "mock":
            from ..baseline.mockfit import write_mock_supervised
            write_mock_supervised(ds, MODEL)
            continue
        rec = aws_cli.succeeded(n, MODEL)
        if rec is None:
            raise RuntimeError(f"{n} has no successful BERT execution; run `encoder run` first")
        bucket, key = rec["steps"]["Train"]["output"].removeprefix("s3://").split("/", 1)
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "output.tar.gz"
            make(rec["region"]).download(bucket, key, tar)
            with tarfile.open(tar) as t:
                files = {m.name.lstrip("./"): t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        job = json.loads(files["run.json"])
        r0 = job["repeats"]["0"]
        run = {**job, "model_version": f"{cfg['model_id']}@{cfg['revision'][:12]}",
               "label_budget": [job["repeats"][str(r)]["label_budget"] for r in range(len(ds.repeats))],
               "region": rec["region"], "instance_type": rec["instance_type"], "execution_arn": rec["execution_arn"],
               "jobs": {k: {x: v.get(x) for x in ("job", "image", "billable_seconds")} for k, v in rec["steps"].items()},
               "billable_seconds": rec["billable_seconds"], "cost_usd": rec["cost_usd"], "fetched_at": aws_cli.now(),
               "infer_s_per_item": r0["infer_s_per_item"]}
        ingest(ds, files, run)
        print(f"encoder fetch: {n}: predictions for {len(ds.repeats)} repeat(s) saved; identifiers match the frozen ones")
    return 0


def run(args) -> int:
    if args.action == "plan":
        return plan(args)
    if args.action == "fetch":
        return fetch(args)
    if args.mode != "live":
        print("encoder run: nothing to do in mock mode (mock BERT is written by `predict --mode mock`)")
        return 0
    return run_live(args)
