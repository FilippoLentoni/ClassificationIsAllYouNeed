"""`baseline plan | deploy | run | fetch | teardown` (design D6).

`run` needs a working instance type and region. If none is recorded yet it probes with the
smallest dataset, moving through the configured instance types and then regions without
asking, and records every attempt. All datasets then run on the combination that worked.
"""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import backup, config, templates
from ..data import datasets
from ..data.datasets import Dataset
from ..results import model_dir, prob_columns, write_xgb_predictions
from . import package, pipeline

REFUSED = ("ResourceLimitExceeded", "service limit", "account-level", "CapacityError", "capacity",
           "InsufficientInstanceCapacity", "not supported", "not available", "is not authorized to perform: sagemaker:Create")
SCRIPTS = Path(__file__).parent / "scripts"
TEMPLATE = config.ROOT / "infra" / "sagemaker.yaml"


def make_aws(region: str):
    from .aws import Aws

    return Aws(region)


def new_run_id() -> str:
    """UTC time plus a random suffix: two executions started in the same second never share an id."""
    import secrets
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(2)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _live(name: str) -> Path:
    return config.results_dir("live") / name


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def stack_state() -> dict:
    return read_json(_live("aws_stack.json"), {})


def template_sha() -> str:
    import hashlib
    return hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


def attempts(model: str | None = None) -> list:
    """Every SageMaker execution attempt of the study, for all models (XGBoost, BERT, Qwen) in one record.

    Records written before BERT and Qwen existed have no `model` field; they are XGBoost's.
    """
    recs = [{"model": "xgb", **a} for a in read_json(_live("baseline_attempts.json"), [])]
    return [a for a in recs if model is None or a["model"] == model]


def record_attempt(rec: dict) -> None:
    if "model" not in rec:
        raise ValueError("an attempt record must name its model")
    _live("baseline_attempts.json").write_text(json.dumps(attempts() + [rec], indent=1, default=str))


def recorded_cost() -> float:
    """Cost of every recorded execution of every model: the spending limit covers all SageMaker work."""
    return float(sum(a.get("cost_usd", 0.0) for a in attempts()))


def refused(reason: str | None) -> bool:
    """True when a failure means this instance type or region cannot be used (not a fault in our code)."""
    return bool(reason) and any(k.lower() in reason.lower() for k in REFUSED)


# ---- plan / deploy / teardown ------------------------------------------------------------------

def plan(args) -> int:
    cfg = config.load("baseline")
    names = templates.selected(args)
    region, itype = cfg["regions"][0], cfg["instance_types"][0]
    print(f"baseline plan: template {TEMPLATE.relative_to(config.ROOT)} creates, per region used: one S3 bucket "
          f"(ciayn-baseline-<account>-<region>), one role (ciayn-sagemaker-exec-<region>), one policy attached to "
          f"{cfg['operator_role']} (ciayn-sagemaker-operator-<region>)")
    print(f"baseline plan: deploy with `python -m ciayn baseline deploy --mode live --yes` (after the user approves the template)")
    print(f"baseline plan: tried in order, instance types {cfg['instance_types']} in regions {cfg['regions']}")
    for n in names:
        print(f"baseline plan: {n}: 1 pipeline execution = 1 processing job + 1 training job "
              f"(worst case ${pipeline.worst_case_cost(itype, region):.2f} on {itype})")
    print(f"baseline plan: spending limit ${config.load('budgets')['aws_spend_limit_usd']:.2f}; recorded so far ${recorded_cost():.2f}; "
          "nothing was created or started")
    return 0


def deploy_region(region: str, aws=None) -> dict:
    cfg = config.load("baseline")
    aws = aws or make_aws(region)
    out = aws.stack_outputs(cfg["stack_name"]) or aws.deploy_stack(cfg["stack_name"], TEMPLATE.read_text(), cfg["operator_role"])
    state = stack_state()
    state.setdefault("stacks", {})[region] = {"stack": cfg["stack_name"], "bucket": out["BucketName"],
                                              "role_arn": out["ExecutionRoleArn"], "deployed_at": now(),
                                              "template_sha256": template_sha()}
    state.setdefault("region", region)
    state.setdefault("bucket", out["BucketName"])
    _live("aws_stack.json").write_text(json.dumps(state, indent=1))
    return state["stacks"][region]


def deploy(args) -> int:
    if not args.yes:
        raise RuntimeError("baseline deploy creates AWS resources; show the user infra/sagemaker.yaml and pass --yes "
                           "only after they approve it")
    state = stack_state()
    state["approved_at"] = state.get("approved_at") or now()
    _live("aws_stack.json").write_text(json.dumps(state, indent=1))
    region = config.load("baseline")["regions"][0]
    s = deploy_region(region)
    print(f"baseline deploy: stack ready in {region}: bucket {s['bucket']}, role {s['role_arn']}")
    return 0


def update(args) -> int:
    """Apply an approved change of infra/sagemaker.yaml to every region where the stack exists."""
    if not args.yes:
        raise RuntimeError("baseline update changes AWS resources; show the user the change to infra/sagemaker.yaml "
                           "and pass --yes only after they approve it")
    cfg, state = config.load("baseline"), stack_state()
    for region, s in state.get("stacks", {}).items():
        out = make_aws(region).update_stack(s["stack"], TEMPLATE.read_text(), cfg["operator_role"])
        s.update(bucket=out["BucketName"], role_arn=out["ExecutionRoleArn"], updated_at=now(), template_sha256=template_sha())
        print(f"baseline update: stack {s['stack']} in {region} matches infra/sagemaker.yaml ({s['template_sha256'][:12]})")
    state.setdefault("template_approvals", []).append({"sha256": template_sha(), "approved_at": now()})
    _live("aws_stack.json").write_text(json.dumps(state, indent=1))
    return 0


def ensure_stack(region: str, aws) -> dict:
    state = stack_state()
    if not state.get("approved_at"):
        raise RuntimeError("the AWS template has not been approved and deployed; run `baseline deploy` first")
    s = state.get("stacks", {}).get(region)
    if s is None:
        return deploy_region(region, aws)
    if s.get("template_sha256") != template_sha():
        raise RuntimeError(f"infra/sagemaker.yaml differs from the template deployed in {region}; show the change to the "
                           "user and run `baseline update --yes` after they approve it")
    return s


def pipeline_names() -> list[str]:
    return [config.load(c)["pipeline_name"] for c in ("baseline", "encoder", "llm")]


def teardown(args) -> int:
    if not args.yes:
        raise RuntimeError("baseline teardown deletes the bucket, role and policy; pass --yes once the user confirms")
    ok, why = backup.local_matches_latest()
    if not ok:
        raise RuntimeError(f"teardown refused: {why}. Run `python -m ciayn backup --mode live` and check the local "
                           "cache, ledger and exchange log first")
    cfg, state = config.load("baseline"), stack_state()
    for region, s in list(state.get("stacks", {}).items()):
        aws = make_aws(region)
        for name in pipeline_names():
            aws.delete_pipeline(name)
        n = aws.empty_bucket(s["bucket"])
        aws.delete_stack(s["stack"])
        print(f"baseline teardown: {region}: pipeline deleted, {n} objects removed, stack {s['stack']} deleted")
        state["stacks"].pop(region)
    state.update(torn_down_at=now(), bucket=None)
    _live("aws_stack.json").write_text(json.dumps(state, indent=1))
    return 0


# ---- run ---------------------------------------------------------------------------------------

def check_spend(itype: str, region: str, worst_case_usd: float | None = None) -> None:
    limit = float(config.load("budgets")["aws_spend_limit_usd"])
    need = pipeline.worst_case_cost(itype, region) if worst_case_usd is None else float(worst_case_usd)
    if recorded_cost() + need > limit:
        raise RuntimeError(f"AWS spending limit: ${recorded_cost():.2f} recorded + ${need:.2f} worst case for the next "
                           f"execution on {itype} exceeds the limit of ${limit:.2f}; no job was started")


def running_marker(name: str, model: str = "xgb") -> Path:
    return _live(f"baseline_running_{name}.json" if model == "xgb" else f"{model}_running_{name}.json")


def execute(ds: Dataset, region: str, itype: str, aws, stack: dict) -> dict:
    """One pipeline execution for a dataset. Returns the attempt record (also appended to the attempts file)."""
    cfg = config.load("baseline")
    check_spend(itype, region)
    running_marker(ds.name).write_text(json.dumps({"dataset": ds.name, "started": now()}))
    try:
        return _execute(ds, region, itype, aws, stack, cfg)
    finally:
        running_marker(ds.name).unlink(missing_ok=True)


def _execute(ds: Dataset, region: str, itype: str, aws, stack: dict, cfg: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = package.write_inputs(ds, Path(tmp))
        aws.upload_dir(root, stack["bucket"], f"input/{ds.name}")
    aws.upload_dir(SCRIPTS, stack["bucket"], "code")
    definition = pipeline.definition(region, stack["bucket"], stack["role_arn"])
    return run_pipeline("xgb", ds.name, region, itype, aws, stack, cfg["pipeline_name"], definition, {"Dataset": ds.name})


def run_pipeline(model: str, dataset: str, region: str, itype: str, aws, stack: dict, pipeline_name: str,
                 definition: dict, params: dict, instance_types: dict | None = None, run_id: str | None = None) -> dict:
    """Create or update a pipeline, run one execution to its end, and record it with its cost.

    `instance_types` gives the instance type of each step when steps use different ones (Qwen's CPU
    staging step and GPU inference step); otherwise every step ran on `itype`.
    """
    aws.upsert_pipeline(pipeline_name, definition, stack["role_arn"])
    run_id = run_id or new_run_id()
    rec = {"model": model, "dataset": dataset, "region": region, "instance_type": itype, "run_id": run_id, "started": now()}
    try:
        arn = aws.start(pipeline_name, {**params, "RunId": run_id, "InstanceType": itype})
        rec["execution_arn"] = arn
        print(f"{model} run: {dataset} on {itype} in {region}: started {arn.split('/')[-1]}", flush=True)
        rec.update(aws.wait(arn, capacity_timeout_s=config.load("baseline").get("capacity_wait_s")))
    except Exception as e:                                # start itself refused (permissions, limits)
        rec.update(status="NotStarted", failure=f"{type(e).__name__}: {e}", steps={})
    types = instance_types or {}
    for name, s in rec["steps"].items():
        s["instance_type"] = types.get(name, itype)
        s["cost_usd"] = pipeline.cost(s["instance_type"], region, s.get("billable_seconds", 0.0))
    seconds = sum(s.get("billable_seconds", 0.0) for s in rec["steps"].values())
    rec.update(ended=now(), billable_seconds=seconds, cost_usd=float(sum(s["cost_usd"] for s in rec["steps"].values())),
               reason=rec.get("failure") or next((s["failure"] for s in rec["steps"].values() if s.get("failure")), None))
    rec["refused"] = rec["status"] != "Succeeded" and refused(rec["reason"])
    record_attempt(rec)
    return rec


def quota_blocks(aws, itype: str) -> str | None:
    for kind in ("processing", "training"):
        q = aws.quota(itype, kind)
        if q is not None and q < 1:
            return f"quota for {itype} {kind} jobs is {q:g}"
    return None


def find_combination(make=make_aws) -> dict:
    """Probe instance types, then regions, until the probe dataset's execution completes."""
    cfg = config.load("baseline")
    combo_path = _live("baseline_combination.json")
    if combo_path.exists():
        return json.loads(combo_path.read_text())
    ds = datasets.load(cfg["probe_dataset"])
    for region in cfg["regions"]:
        aws = make(region)
        stack = ensure_stack(region, aws)
        for itype in cfg["instance_types"]:
            blocked = quota_blocks(aws, itype)
            if blocked:
                rec = {"model": "xgb", "dataset": ds.name, "region": region, "instance_type": itype, "status": "Skipped",
                       "reason": blocked, "refused": True, "cost_usd": 0.0, "started": now()}
                record_attempt(rec)
                print(f"baseline run: {itype} in {region} skipped: {blocked}", flush=True)
                continue
            rec = execute(ds, region, itype, aws, stack)
            if rec["status"] == "Succeeded":
                combo = {"region": region, "instance_type": itype, "found_at": now(), "probe_execution": rec["execution_arn"]}
                combo_path.write_text(json.dumps(combo, indent=1))
                _use_region(region, make)
                return combo
            print(f"baseline run: {itype} in {region} did not complete: {rec['status']}: {rec['reason']}", flush=True)
            if not rec["refused"]:
                raise RuntimeError(f"the {ds.name} execution failed for a reason that is not instance or region "
                                   f"availability, so trying other combinations would not help: {rec['reason']}")
    raise RuntimeError(f"SageMaker refused every configured combination ({len(cfg['regions'])} regions x "
                       f"{len(cfg['instance_types'])} instance types); see results/live/baseline_attempts.json")


def _use_region(region: str, make) -> None:
    """Make `region` the active one and delete stacks in regions that turned out not to be needed."""
    cfg, state = config.load("baseline"), stack_state()
    for other, s in list(state.get("stacks", {}).items()):
        if other != region:
            aws = make(other)
            aws.delete_pipeline(cfg["pipeline_name"])
            aws.empty_bucket(s["bucket"])
            aws.delete_stack(s["stack"])
            state["stacks"].pop(other)
            print(f"baseline run: removed the unused stack in {other}", flush=True)
    state.update(region=region, bucket=state["stacks"][region]["bucket"])
    _live("aws_stack.json").write_text(json.dumps(state, indent=1))


def succeeded(name: str, model: str = "xgb") -> dict | None:
    done = [a for a in attempts(model) if a.get("dataset") == name and a.get("status") == "Succeeded"]
    return done[-1] if done else None


def run_live(args, make=make_aws) -> int:
    cfg = config.load("baseline")
    names = templates.selected(args)
    todo = [n for n in names if args.rerun or not succeeded(n)]
    if args.dry_run:
        itype = cfg["instance_types"][0]
        for n in todo:
            print(f"[dry-run] baseline run: {n}: would start 1 processing job and 1 training job "
                  f"(worst case ${pipeline.worst_case_cost(itype, cfg['regions'][0]):.2f})")
        print(f"[dry-run] baseline run: {len(todo)} executions; recorded cost ${recorded_cost():.2f}; nothing was started")
        return 0
    probing = not _live("baseline_combination.json").exists()
    combo = find_combination(make)
    region, itype = combo["region"], combo["instance_type"]
    aws = make(region)
    stack = ensure_stack(region, aws)
    if probing:       # the probe dataset has just been executed: do not run it twice
        todo = [n for n in todo if n != cfg["probe_dataset"]]
    for n in todo:
        if running_marker(n).exists():      # another process is already running this dataset: do not start it twice
            print(f"baseline run: {n} is already running in another process; skipped", flush=True)
            continue
        rec = execute(datasets.load(n), region, itype, aws, stack)
        if rec["status"] != "Succeeded":
            raise RuntimeError(f"baseline run: {n} {rec['status']}: {rec['reason']}")
        print(f"baseline run: {n} succeeded in {rec['billable_seconds']:.0f} billable seconds (${rec['cost_usd']:.3f})", flush=True)
    print(f"baseline run: {len(names)} datasets done on {itype} in {region}; recorded AWS cost ${recorded_cost():.2f}")
    return 0


# ---- fetch -------------------------------------------------------------------------------------

def check_ids(ds: Dataset, frame: pd.DataFrame) -> None:
    """Refuse predictions whose identifiers differ from the frozen ones."""
    for r, rep in enumerate(ds.repeats):
        for split in ("val", "test"):
            got = frame[(frame.repeat == r) & (frame.split == split)]["id"].tolist()
            if sorted(got) != sorted(rep[split]):
                raise RuntimeError(f"{ds.name}: baseline predictions for repeat {r} {split} do not match the frozen "
                                   f"identifiers ({len(got)} returned, {len(rep[split])} expected)")


def ingest(ds: Dataset, pred_csv, curve_csv, run: dict, mode: str = "live") -> None:
    cols = prob_columns(len(ds.classes))
    frame = pd.read_csv(pred_csv)
    check_ids(ds, frame)
    curve = pd.read_csv(curve_csv)
    for r, rep in enumerate(ds.repeats):
        bad = set(curve[curve.repeat == r]["id"]) - set(rep["test"])
        if bad:
            raise RuntimeError(f"{ds.name}: learning-curve predictions contain identifiers outside the frozen test part")
    write_xgb_predictions(mode, ds, frame[["id", "repeat", "split"] + cols], run)
    curve.to_parquet(model_dir(mode, ds.name, "xgb") / "learning_curve.parquet", index=False)


def fetch(args, make=make_aws) -> int:
    for n in templates.selected(args):
        ds = datasets.load(n)
        if args.mode == "mock":
            from .mockfit import write_mock_baseline

            write_mock_baseline(ds)
            continue
        rec = succeeded(n)
        if rec is None:
            raise RuntimeError(f"{n} has no successful baseline execution; run `baseline run` first")
        bucket, key = rec["steps"]["Train"]["output"].removeprefix("s3://").split("/", 1)
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "output.tar.gz"
            make(rec["region"]).download(bucket, key, tar)
            with tarfile.open(tar) as t:
                files = {m.name.lstrip("./"): t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        job = json.loads(files["run.json"])
        run = {**job, "model_version": f"xgboost-{job['versions']['xgboost']}", "region": rec["region"],
               "instance_type": rec["instance_type"], "execution_arn": rec["execution_arn"],
               "jobs": {k: {x: v.get(x) for x in ("job", "image", "billable_seconds")} for k, v in rec["steps"].items()},
               "billable_seconds": rec["billable_seconds"], "cost_usd": rec["cost_usd"], "fetched_at": now()}
        ingest(ds, io.BytesIO(files["predictions.csv"]), io.BytesIO(files["learning_curve.csv"]), run)
        print(f"baseline fetch: {n}: predictions for {len(ds.repeats)} repeat(s) saved; identifiers match the frozen ones")
    return 0


def run(args) -> int:
    if args.action == "plan":
        return plan(args)
    if args.action == "fetch":
        return fetch(args)
    if args.mode != "live":
        print(f"baseline {args.action}: nothing to do in mock mode (the mock baseline is written by `predict --mode mock`)")
        return 0
    return {"deploy": deploy, "update": update, "run": run_live, "teardown": teardown}[args.action](args)
