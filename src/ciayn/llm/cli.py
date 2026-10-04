"""`llm plan | stage | smoke | run | fetch | project-thinking` (design D3-D5): Qwen3.6-27B on SageMaker.

- `stage` copies the pinned weights to S3 once (CPU processing job only).
- `smoke` runs 20 items per dataset on the GPU, trying the configured instance types and then
  regions in order, and saves a projection of the full run's cost from the measured timings.
- `run` starts the full run only if that projection plus everything already recorded fits the
  spending limit. `--thinking all` additionally needs `project-thinking`, computed from the full
  run's thinking subsample, to fit.
- `fetch` reads the per-dataset results from the job's checkpoint location, so a stopped job
  still yields the datasets it finished, and checks every identifier against Jev's.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, escalation_val, templates
from ..baseline import cli as aws_cli
from ..baseline import pipeline as base
from ..data import datasets
from ..results import prob_columns, write_item_predictions
from . import pipeline, prompts

MODEL = "qwen"
SCRIPTS = Path(__file__).parent / "scripts"


def _p(name: str) -> Path:
    return aws_cli._live(name)


def read(name: str, default=None):
    return json.loads(_p(name).read_text()) if _p(name).exists() else default


def runs(kind: str) -> list[dict]:
    return [a for a in aws_cli.attempts(MODEL) if a.get("kind") == kind and a.get("status") == "Succeeded"]


# ---- plan and stage ----------------------------------------------------------------------------

def plan(args) -> int:
    cfg = config.load("llm")
    region, itype = cfg["infer"]["regions"][0], cfg["infer"]["instance_types"][0]
    print(f"llm plan: {cfg['model_id']} at {cfg['revision'][:12]}, weights staged once to s3://<bucket>/{cfg['weights_prefix']}/")
    print(f"llm plan: smoke worst case ${pipeline.worst_case_cost(itype, region, cfg['smoke']['max_runtime_s']):.2f}; full run "
          f"worst case ${pipeline.worst_case_cost(itype, region, cfg['infer']['max_runtime_s']):.2f} on {itype}")
    proj = read("qwen_projection.json")
    if proj and "projected_usd" in proj:
        print(f"llm plan: projected remaining Qwen jobs ${proj['projected_usd']:.2f}; fits the limit: {proj['fits']}")
    print(f"llm plan: {pipeline.SUBSTITUTION}")
    print(f"llm plan: spending limit ${config.load('budgets')['aws_spend_limit_usd']:.2f}, recorded ${aws_cli.recorded_cost():.2f}")
    return 0


def execute(kind: str, dss: list, region: str, itype: str, aws, stack: dict, max_runtime_s: int, inference: bool = True,
            reasoning: dict | None = None) -> dict:
    aws_cli.check_spend(itype, region, pipeline.worst_case_cost(itype, region, max_runtime_s, inference))
    marker = aws_cli.running_marker(kind, MODEL)
    marker.write_text(json.dumps({"kind": kind, "started": aws_cli.now()}))
    run_id = aws_cli.new_run_id()
    try:
        aws.upload_dir(SCRIPTS, stack["bucket"], "code-qwen")
        if inference:
            with tempfile.TemporaryDirectory() as tmp:
                aws.upload_dir(prompts.write_job(dss, kind, Path(tmp) / "job", reasoning=reasoning), stack["bucket"],
                               f"input-qwen/{run_id}")
        cfg = config.load("llm")
        rec = aws_cli.run_pipeline(MODEL, ",".join(d.name for d in dss) or "weights", region, itype, aws, stack,
                                   cfg["pipeline_name"], pipeline.definition(region, stack["bucket"], stack["role_arn"], max_runtime_s),
                                   {"RunInference": "yes" if inference else "no"},
                                   instance_types={"Stage": cfg["stage"]["instance_type"]}, run_id=run_id)
    finally:
        marker.unlink(missing_ok=True)
    rec.update(kind=kind, substitution=pipeline.SUBSTITUTION if inference else None,
               checkpoints=f"s3://{stack['bucket']}/checkpoints/qwen/{run_id}/")
    recs = [rec if a.get("model") == MODEL and a.get("run_id") == run_id else a for a in aws_cli.attempts()]
    # run_pipeline recorded it; the kind and the checkpoint location are added here
    aws_cli._live("baseline_attempts.json").write_text(json.dumps(recs, indent=1, default=str))
    return rec


def stage(args, make=aws_cli.make_aws) -> int:
    cfg = config.load("llm")
    region = (read("llm_combination.json") or {}).get("region", cfg["infer"]["regions"][0])
    aws = make(region)
    st = aws_cli.ensure_stack(region, aws)
    key = f"{cfg['weights_prefix']}/{cfg['revision']}/MANIFEST.sha256"
    if aws.exists(st["bucket"], key):
        print(f"llm stage: s3://{st['bucket']}/{key} exists; nothing to do")
        return 0
    if args.dry_run:
        print(f"[dry-run] llm stage: would start 1 processing job on {cfg['stage']['instance_type']} to copy the weights")
        return 0
    rec = execute("stage", [], region, cfg["stage"]["instance_type"], aws, st, 0, inference=False)
    if rec["status"] != "Succeeded" or not aws.exists(st["bucket"], key):
        raise RuntimeError(f"llm stage: {rec['status']}: {rec['reason']}")
    print(f"llm stage: weights staged in {rec['billable_seconds']:.0f} billable seconds (${rec['cost_usd']:.3f})")
    return 0


# ---- smoke, projection and the full runs ---------------------------------------------------------

def _combos(cfg):
    """The combination that worked last, then every other one, in the configured order."""
    combo = read("llm_combination.json")
    every = [(r, it) for r in cfg["infer"]["regions"] for it in cfg["infer"]["instance_types"]]
    first = [(combo["region"], combo["instance_type"])] if combo else []
    return first + [c for c in every if c not in first]


def recently_refused(cfg) -> set:
    from datetime import datetime, timezone
    now, out = datetime.now(timezone.utc), set()
    for a in aws_cli.attempts(MODEL):
        when = a.get("ended") or a.get("started")
        if a.get("refused") and when:
            age = (now - datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()
            if age < cfg["infer"]["retry_refused_after_s"]:
                out.add((a["region"], a["instance_type"]))
    return out


def execute_somewhere(kind: str, dss: list, max_runtime_s: int, args, make, reasoning: dict | None = None) -> dict:
    """Run one job on the first combination that has capacity; every attempt is recorded."""
    cfg = config.load("llm")
    refused = recently_refused(cfg)
    for region, itype in _combos(cfg):
        if (region, itype) in refused and not getattr(args, "rerun", False):
            print(f"llm {kind}: {itype} in {region} was refused in the last {cfg['infer']['retry_refused_after_s'] // 3600} h; skipped", flush=True)
            continue
        worst = pipeline.worst_case_cost(itype, region, max_runtime_s)
        limit = float(config.load("budgets")["aws_spend_limit_usd"])
        if aws_cli.recorded_cost() + worst > limit:
            print(f"llm {kind}: {itype} in {region} skipped: worst case ${worst:.2f} plus ${aws_cli.recorded_cost():.2f} recorded "
                  f"exceeds the ${limit:.2f} limit", flush=True)
            continue
        aws = make(region)
        st = aws_cli.ensure_stack(region, aws)
        q = aws.quota(itype, "training")
        if q is not None and q < 1:
            aws_cli.record_attempt({"model": MODEL, "kind": kind, "dataset": "all", "region": region, "instance_type": itype,
                                    "status": "Skipped", "reason": f"quota for {itype} training jobs is {q:g}", "refused": True,
                                    "cost_usd": 0.0, "started": aws_cli.now(), "ended": aws_cli.now()})
            print(f"llm {kind}: {itype} in {region} skipped: training-job quota {q:g}", flush=True)
            continue
        rec = execute(kind, dss, region, itype, aws, st, max_runtime_s, reasoning=reasoning)
        if rec["status"] == "Succeeded":
            _p("llm_combination.json").write_text(json.dumps({"region": region, "instance_type": itype, "found_at": aws_cli.now(),
                                                              "probe_execution": rec["execution_arn"]}, indent=1))
            return rec
        print(f"llm {kind}: {itype} in {region} did not complete: {rec['status']}: {rec['reason']}", flush=True)
        if not rec["refused"]:
            raise RuntimeError(f"the Qwen {kind} job failed for a reason that is not instance or region availability: {rec['reason']}")
    raise RuntimeError("SageMaker refused every configured combination for Qwen; see results/live/baseline_attempts.json")


def _datasets(kind: str) -> list:
    return [datasets.load(n) for n in config.DATASETS if any(prompts.sets(datasets.load(n), kind).values())]


def smoke(args, make=aws_cli.make_aws) -> int:
    """Measure seconds per item. `--engine vllm`: the datasets vLLM scores; `--engine transformers`: the datasets it scores."""
    cfg = config.load("llm")
    engine = getattr(args, "engine", None) or "vllm"
    kind = "smoke" if engine == "vllm" else "smoke-hf"
    dss = _datasets(kind)
    if args.dry_run:
        it = cfg["infer"]["instance_types"][0]
        print(f"[dry-run] llm smoke ({engine}): {', '.join(d.name for d in dss)}; worst case "
              f"${pipeline.worst_case_cost(it, cfg['infer']['regions'][0], cfg['smoke']['max_runtime_s']):.2f} on {it}")
        return 0
    done = runs(kind)
    rec = done[-1] if done and not args.rerun else execute_somewhere(kind, dss, cfg["smoke"]["max_runtime_s"], args, make)
    run = json.loads(download_checkpoints(rec, make)["run.json"])
    failed = {n: i["nothink"]["invalid"] for n, i in run["datasets"].items() if i.get("nothink", {}).get("invalid")}
    if failed and engine == "transformers":
        raise RuntimeError(f"the smoke run could not score some items ({failed}); no projection is made from a run with failures")
    proj = read("qwen_projection.json") or {}
    if "engine" in proj:                  # the first projection had one engine only; keep it for the record
        proj = {"superseded": proj}
    proj[engine] = project(run, rec, "full" if engine == "vllm" else "full-hf")
    proj.update(_totals(proj))
    _p("qwen_projection.json").write_text(json.dumps(proj, indent=1))
    e = proj[engine]
    print(f"llm smoke ({engine}): measured on {rec['instance_type']} in {rec['region']}; projected {e['kind']} job "
          f"{e['projected_hours']:.2f} h, ${e['projected_usd']:.2f}; both jobs ${proj['projected_usd']:.2f}; fits the limit: {proj['fits']}")
    return 0


def _per_item(info: dict, variant: str) -> float:
    v = info.get(variant)
    return v["seconds"] / v["items"] if v and v.get("items") else 0.0


def project(run: dict, rec: dict, kind: str) -> dict:
    """Billable time of the `kind` job, from a smoke run's measured seconds per item (same engine)."""
    cfg = config.load("llm")
    measured = run["load_s"] + sum(i.get("nothink", {}).get("seconds", 0) + i.get("think", {}).get("seconds", 0) +
                                   sum(i.get("latency_s", [])) for i in run["datasets"].values())
    infer_billable = rec["steps"].get("Infer", {}).get("billable_seconds", 0.0)
    overhead = max(infer_billable - measured, 0.0)          # instance start, weight copy, container start
    secs, per_ds = overhead + run["load_s"], {}
    for ds in _datasets(kind):
        info = run["datasets"].get(ds.name)
        s = prompts.sets(ds, kind)
        if info is None:                       # this smoke run did not measure the dataset (reasoning-only in the full job)
            per_ds[ds.name] = {"not_measured": True}
            continue
        lat = float(np.mean(info["latency_s"])) if info.get("latency_s") else 0.0
        nt, th = _per_item(info, "nothink"), _per_item(info, "think")
        if prompts.think_mode(ds, kind) == "generate":
            th = max(th - nt, 0.0)             # reasoning only: the scoring part runs in the other job
        if prompts.think_mode(ds, kind) == "given":
            th = nt                            # scoring after supplied reasoning costs about one scoring pass
        t = nt * len(s["nothink"]) + th * len(s["think"]) + lat * len(s["latency"])
        per_ds[ds.name] = {"nothink_s_per_item": nt, "think_s_per_item": th, "seconds": t}
        secs += t
    itype, region, secs = rec["instance_type"], rec["region"], float(secs)
    usd = float(base.cost(itype, region, secs) + base.cost(cfg["stage"]["instance_type"], region, 120))
    return {"kind": kind, "engine": run["engine"], "instance_type": itype, "region": region, "projected_seconds": secs,
            "projected_hours": secs / 3600, "projected_usd": usd, "overhead_s": overhead, "load_s": run["load_s"],
            "datasets": per_ds, "from_execution": rec["execution_arn"], "made_at": aws_cli.now(),
            "note": "an upper estimate where the smoke run used larger thinking settings than the full job"}


def _totals(proj: dict) -> dict:
    cfg = config.load("llm")
    parts = [proj[e] for e in ("vllm", "transformers") if e in proj]
    done = {r["kind"] for r in aws_cli.attempts(MODEL) if r.get("status") == "Succeeded"}
    todo = [p for p in parts if p["kind"] not in done]
    usd, limit = float(sum(p["projected_usd"] for p in todo)), float(config.load("budgets")["aws_spend_limit_usd"])
    return {"projected_usd": usd, "recorded_usd": aws_cli.recorded_cost(), "limit_usd": limit,
            "complete": len(parts) == 2,
            "fits": bool(len(parts) == 2 and aws_cli.recorded_cost() + usd <= limit and
                         all(p["projected_seconds"] <= 0.9 * cfg["infer"]["max_runtime_s"] for p in todo))}


def project_thinking(args, make=aws_cli.make_aws) -> int:
    full = runs("full")
    if not full:
        raise RuntimeError("project-thinking needs a completed full run (it measures the thinking subsample)")
    rec = full[-1]
    run = json.loads(download_checkpoints(rec, make)["run.json"])
    secs = run["load_s"]
    for n in config.DATASETS:
        info = run["datasets"].get(n) or {}
        secs += _per_item(info, "think") * len(datasets.load(n).jev_ids)
    secs = float(secs)
    usd = float(base.cost(rec["instance_type"], rec["region"], secs * 1.1))      # 10% for instance start and weight copy
    scoring = runs("full-hf")                       # scoring after the reasoning costs about one scoring pass over every item
    score_usd = 0.0
    if scoring:
        srun = json.loads(download_checkpoints(scoring[-1], make)["run.json"])
        score_s = srun["load_s"] + sum(_per_item(i, "nothink") * len(datasets.load(n).jev_ids) for n, i in srun["datasets"].items())
        score_usd = float(base.cost(scoring[-1]["instance_type"], scoring[-1]["region"], score_s * 1.1))
    reasoning_usd, usd = usd, usd + score_usd
    limit = float(config.load("budgets")["aws_spend_limit_usd"])
    margin = float(config.load("llm")["thinking"]["projection_margin"])      # a projection is not a bill
    fits = bool(aws_cli.recorded_cost() + usd * (1 + margin) <= limit)
    proj = {"projected_seconds": secs * 1.1, "projected_usd": usd, "reasoning_usd": reasoning_usd, "scoring_usd": score_usd,
            "recorded_usd": aws_cli.recorded_cost(), "limit_usd": limit, "margin": margin,
            "fits": fits, "from_execution": rec["execution_arn"], "made_at": aws_cli.now(),
            "basis": "reasoning: seconds per item of the thinking subsample in the vLLM job, times every item Jev predicted; "
                     "scoring: one scoring pass over those items at the measured seconds per item of the transformers job",
            "decision": "run thinking enabled on every item" if fits else "thinking enabled stays on the subsample: the projection does not fit the limit"}
    _p("qwen_thinking_projection.json").write_text(json.dumps(proj, indent=1))
    print(f"llm project-thinking: every item with thinking enabled would take {secs * 1.1 / 3600:.1f} h of reasoning, ${usd:.2f} with scoring; "
          f"recorded ${aws_cli.recorded_cost():.2f}, limit ${limit:.2f}: {proj['decision']}")
    return 0


def saved_reasoning(make=aws_cli.make_aws) -> dict:
    """The reasoning the vLLM full job generated for the datasets the transformers engine scores."""
    full = runs("full")
    if not full:
        return {}
    out = {}
    for name, data in download_checkpoints(full[-1], make).items():
        if name.endswith(".think_reasoning.jsonl"):
            rows = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
            out[name.split(".")[0]] = {r["id"]: r for r in rows}
    return out


def run_live(args, make=aws_cli.make_aws) -> int:
    cfg = config.load("llm")
    if args.thinking == "all":
        proj = read("qwen_thinking_projection.json")
        if proj is None:
            raise RuntimeError("run `llm project-thinking` first")
        remaining = float(config.load("budgets")["aws_spend_limit_usd"]) - aws_cli.recorded_cost()
        if not proj["fits"] or proj["projected_usd"] > remaining:
            raise RuntimeError(f"llm run refused: thinking on every item is projected at ${proj['projected_usd']:.2f}, but only "
                               f"${remaining:.2f} of the limit remains")
        kinds = ["thinking-all"]
    else:
        kinds = ["full", "full-hf"]
    proj = read("qwen_projection.json") or {}
    for kind in kinds:
        if runs(kind) and not args.rerun:
            print(f"llm run: the {kind} job already succeeded; use --rerun to run it again")
            continue
        if kind in ("full", "full-hf"):
            proj.update(_totals(proj))
            part = proj.get(prompts.KINDS[kind])
            if part is None or not proj.get("complete"):
                raise RuntimeError("no complete cost projection yet: run `llm smoke` and `llm smoke --engine transformers` first")
            if not proj["fits"]:
                raise RuntimeError(f"llm run refused: the remaining Qwen jobs are projected at ${proj['projected_usd']:.2f}; with "
                                   f"${proj['recorded_usd']:.2f} recorded this exceeds the ${proj['limit_usd']:.2f} limit or a job's time limit")
            max_rt = int(min(cfg["infer"]["max_runtime_s"], max(1800, 1.5 * part["projected_seconds"])))
        else:
            max_rt = cfg["infer"]["max_runtime_s"]
        if args.dry_run:
            print(f"[dry-run] llm run: {kind}: {', '.join(d.name for d in _datasets(kind))}; time limit {max_rt}s; nothing was started")
            continue
        reasoning = saved_reasoning(make) if kind == "full-hf" else None
        rec = execute_somewhere(kind, _datasets(kind), max_rt, args, make, reasoning=reasoning)
        print(f"llm run: {kind} succeeded in {rec['billable_seconds']:.0f} billable seconds (${rec['cost_usd']:.2f})", flush=True)
    return 0


# ---- fetch -------------------------------------------------------------------------------------

def download_checkpoints(rec: dict, make=aws_cli.make_aws) -> dict[str, bytes]:
    bucket, prefix = rec["checkpoints"].removeprefix("s3://").split("/", 1)
    aws, files = make(rec["region"]), {}
    with tempfile.TemporaryDirectory() as tmp:
        for key in aws.list_keys(bucket, prefix):
            path = Path(tmp) / key.rsplit("/", 1)[-1]
            aws.download(bucket, key, path)
            files[path.name] = path.read_bytes()
    return files


def ingest(ds, rows: list[dict], model: str, expected: list[str], run: dict, mode: str = "live") -> pd.DataFrame:
    got = [r["id"] for r in rows]
    if sorted(got) != sorted(expected):
        raise RuntimeError(f"{ds.name}: {model} predictions do not match the expected identifiers "
                           f"({len(got)} returned, {len(expected)} expected)")
    cols = prob_columns(len(ds.classes))
    frame = pd.DataFrame([{"id": r["id"], "valid": r["valid"], **dict(zip(cols, r["p"])), "seconds": r["seconds"],
                           "prompt_tokens": r["prompt_tokens"], **({"thinking_tokens": r["thinking_tokens"],
                                                                     "thinking_truncated": r["thinking_truncated"]}
                                                                    if "thinking_tokens" in r else {})} for r in rows])
    return write_item_predictions(mode, ds, model, frame, run)


def _latest(kind: str) -> dict | None:
    """The latest job of `kind` that has results: a succeeded one, else a stopped one (its finished datasets are kept)."""
    done = runs(kind) or [a for a in aws_cli.attempts(MODEL) if a.get("kind") == kind and a.get("checkpoints") and a.get("billable_seconds")]
    return done[-1] if done else None


def fetch(args, make=aws_cli.make_aws) -> int:
    from . import thinking_subsample
    if args.mode == "mock":
        from ..baseline.mockfit import write_mock_zero_shot
        for n in templates.selected(args):
            ds = datasets.load(n)
            write_mock_zero_shot(ds, "qwen", prompts.all_items(ds))
            write_mock_zero_shot(ds, "qwen_think", thinking_subsample(ds))
        return 0
    cfg = config.load("llm")
    jobs = {}
    for kind in ("full", "full-hf", "thinking-all"):
        rec = _latest(kind)
        if rec:
            files = download_checkpoints(rec, make)
            jobs[kind] = (rec, files, json.loads(files["run.json"]), json.loads(files.get("rendered_examples.json", b"{}")))
    if not jobs:
        raise RuntimeError("no Qwen job has results yet; run `llm run` first")
    for n in templates.selected(args):
        ds = datasets.load(n)
        kind = "full" if prompts.engine_of(n) == "vllm" else "full-hf"
        name = f"{n}.nothink.jsonl"
        if kind not in jobs or name not in jobs[kind][1]:
            print(f"llm fetch: {n}: no results yet from the {kind} job")
            continue
        rec, files, run, rendered = jobs[kind]
        info = run["datasets"][n]
        totals = {"nothink_items": sum(d.get("nothink", {}).get("items", 0) for d in run["datasets"].values()),
                  "nothink_seconds": sum(d.get("nothink", {}).get("seconds", 0) for d in run["datasets"].values()),
                  "think_seconds": sum(d.get("think", {}).get("seconds", 0) for d in run["datasets"].values()),
                  "latency_seconds": sum(sum(d.get("latency_s", [])) for d in run["datasets"].values())}
        meta = {"model_version": f"{cfg['model_id']}@{cfg['revision'][:12]}", "engine": run["engine"], "scoring": run.get("scoring"),
                "versions": run.get("versions"), "gpus": run.get("gpus"), "region": rec["region"],
                "instance_type": rec["instance_type"], "execution_arn": rec["execution_arn"], "substitution": rec.get("substitution"),
                "job_status": rec["status"], "billable_seconds": rec["billable_seconds"], "cost_usd": rec["cost_usd"],
                "load_s": run["load_s"], "timing": info, "rendered_example": rendered.get(n), "cache_stats": run.get("cache_stats"),
                "jobs": {k: {x: v.get(x) for x in ("job", "image", "billable_seconds", "instance_type", "cost_usd")}
                         for k, v in rec["steps"].items() if v.get("job")},
                "run_totals": totals, "fetched_at": aws_cli.now()}
        rows = [json.loads(x) for x in files[name].decode().splitlines() if x.strip()]
        ingest(ds, rows, "qwen", prompts.all_items(ds), meta)
        tname = f"{n}.think.jsonl"
        if "thinking-all" in jobs and tname in jobs["thinking-all"][1]:
            rows = [json.loads(x) for x in jobs["thinking-all"][1][tname].decode().splitlines() if x.strip()]
            ingest(ds, rows, "qwen_think", list(ds.jev_ids), {**meta, "thinking": "every item"})
        elif tname in files:
            rows = [json.loads(x) for x in files[tname].decode().splitlines() if x.strip()]
            gen = jobs.get("full", (None, None, {"datasets": {}}))[2]["datasets"].get(n, {}).get("think")
            ingest(ds, rows, "qwen_think", thinking_subsample(ds), {**meta, "thinking": "seeded subsample",
                                                                    "reasoning_generated_by": "vllm", "reasoning_timing": gen})
        print(f"llm fetch: {n}: {len(rows)} thinking and {len(prompts.all_items(ds))} thinking-disabled predictions saved; "
              f"identifiers match Jev's")
    return 0


def run(args) -> int:
    if args.action == "plan":
        return plan(args)
    if args.action == "fetch":
        return fetch(args)
    if args.mode != "live":
        print(f"llm {args.action}: nothing to do in mock mode (mock Qwen is written by `predict --mode mock`)")
        return 0
    return {"stage": stage, "smoke": smoke, "run": run_live, "project-thinking": project_thinking}[args.action](args)
