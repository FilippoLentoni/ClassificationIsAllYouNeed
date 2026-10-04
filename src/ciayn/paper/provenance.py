"""Actual usage and cost of the study, computed from the ledger, the request log and the job records.

Jev: every request is in the ledger. Token counts exist for the requests in the request log
and, for the calls made before logging, for the responses that were cached. A request with
neither is reported as such, not estimated. AWS: billable seconds of every recorded pipeline
execution at list price; this is an estimate from job records, not the AWS bill.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .. import config
from ..backends import exchange
from ..backends.budget import Ledger

STAGES = ("smoke", "template", "predict", "explain", "escalation_val")


def jev_usage() -> dict:
    price = config.load("jev")["usd_per_mtok_input"]
    ledger = Ledger()
    rows = ledger.db.execute("SELECT n, ts, stage FROM jev_ledger ORDER BY n").fetchall()
    by_stage = {s: {"requests": 0, "logged": 0, "failed": 0, "input_tokens": 0, "latency_sum": 0.0} for s in STAGES}
    for _, _, stage in rows:
        by_stage.setdefault(stage, {"requests": 0, "logged": 0, "failed": 0, "input_tokens": 0, "latency_sum": 0.0})["requests"] += 1
    logged_n, logged_keys = set(), set()
    for path in sorted(exchange.log_dir().glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                s = by_stage[r["stage"]]
                s["logged"] += 1
                logged_n.add(r.get("ledger_n"))
                logged_keys.add(r.get("cache_key"))
                if r.get("status") == 200 and isinstance(r.get("response"), dict):
                    s["input_tokens"] += int(r["response"].get("usage", {}).get("input_tokens", 0))
                    s["latency_sum"] += float(r.get("latency_s", 0.0))
                else:
                    s["failed"] += 1
    # calls made before request logging: token counts survive only for the cached responses
    db = sqlite3.connect(config.CACHE / "responses.sqlite")
    hist_tokens, hist_n = 0, 0
    for key, payload in db.execute("SELECT key, payload FROM responses WHERE backend = 'jev'"):
        if key not in logged_keys:
            hist_tokens += int(json.loads(payload)["meta"].get("input_tokens") or 0)
            hist_n += 1
    by_stage["predict"]["input_tokens"] += hist_tokens     # every cached response from before the log was a prediction
    total = len(rows)
    logged = sum(s["logged"] for s in by_stage.values())
    ok = sum(s["logged"] - s["failed"] for s in by_stage.values())
    tokens = sum(s["input_tokens"] for s in by_stage.values())
    days = sorted({datetime.fromtimestamp(ts, timezone.utc).date().isoformat() for _, ts, _ in rows})
    for s in by_stage.values():
        s["usd"] = s["input_tokens"] * price / 1e6
    return {
        "requests_in_ledger": total, "cap": ledger.cap, "by_stage": by_stage,
        "requests_in_log": logged, "failed_requests": sum(s["failed"] for s in by_stage.values()),
        "requests_before_logging": total - len(logged_n & {n for n, _, _ in rows}),
        "requests_before_logging_with_token_counts": hist_n, "input_tokens_before_logging": hist_tokens,
        "requests_with_token_counts": ok + hist_n, "requests_without_token_counts": total - ok - hist_n,
        "input_tokens": tokens, "usd_per_million_input_tokens": price, "usd": tokens * price / 1e6,
        "mean_latency_s": (sum(s["latency_sum"] for s in by_stage.values()) / ok) if ok else None,
        "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
        "sdk_version": exchange.sdk_version(), "model": config.load("jev")["model"],
    }


COMBINATION = {"xgb": "baseline_combination.json", "bert": "encoder_combination.json", "qwen": "llm_combination.json"}


def aws_usage() -> dict:
    """Totals over every SageMaker attempt of the study, and the same figures for each model."""
    path = config.RESULTS / "live" / "baseline_attempts.json"
    attempts = [{"model": "xgb", **a} for a in (json.loads(path.read_text()) if path.exists() else [])]
    out = _usage(attempts, "xgb")
    out["by_model"] = {m: _usage([a for a in attempts if a["model"] == m], m) for m in COMBINATION}
    return out


def _usage(attempts: list, model: str) -> dict:
    done = [a for a in attempts if a.get("status") == "Succeeded"]
    combo_path = config.RESULTS / "live" / COMBINATION[model]
    combo = json.loads(combo_path.read_text()) if combo_path.exists() else {}
    images = sorted({s.get("image") for a in done for s in a.get("steps", {}).values() if s.get("image")})
    started = [a for a in attempts if a.get("execution_arn")]
    return {"attempts": len(attempts), "refused_attempts": sum(1 for a in attempts if a.get("refused")),
            "executions_started": len(started), "executions_succeeded": len(done),
            "executions_not_completed": len(started) - len(done),
            "executions_superseded": len(done) - len({(a.get("kind"), a["dataset"]) for a in done}),
            "datasets": sorted({a["dataset"] for a in done}),
            "gpu_billable_seconds": float(sum(s.get("billable_seconds", 0.0) for a in attempts for s in a.get("steps", {}).values()
                                              if str(s.get("instance_type", a.get("instance_type", ""))).startswith(("ml.g", "ml.p")))),
            "jobs": sum(1 for a in attempts for s in a.get("steps", {}).values() if s.get("job")), "billable_seconds": float(sum(a.get("billable_seconds", 0.0) for a in attempts)),
            "usd": float(sum(a.get("cost_usd", 0.0) for a in attempts)),
            "spend_limit_usd": float(config.load("budgets")["aws_spend_limit_usd"]),
            "instance_type": combo.get("instance_type"), "region": combo.get("region"), "images": images,
            "basis": "billable seconds of each job at on-demand list price (configs/baseline.yaml); not the AWS bill"}


def compute(mode: str) -> dict:
    if mode != "live":
        zero_aws = {"attempts": 0, "refused_attempts": 0, "executions_started": 0, "executions_succeeded": 0,
                    "executions_not_completed": 0, "executions_superseded": 0, "datasets": [], "jobs": 0,
                    "billable_seconds": 0.0, "gpu_billable_seconds": 0.0, "usd": 0.0,
                    "spend_limit_usd": float(config.load("budgets")["aws_spend_limit_usd"]),
                    "instance_type": "none (mock)", "region": "none (mock)", "images": [], "basis": "mock run: no job was started"}
        return {"mode": mode,
                "jev": {"requests_in_ledger": 0, "cap": int(config.load("budgets")["jev_total"]),
                        "by_stage": {s: {"requests": 0, "logged": 0, "failed": 0, "input_tokens": 0, "usd": 0.0} for s in STAGES},
                        "requests_in_log": 0, "failed_requests": 0, "requests_before_logging": 0,
                        "requests_before_logging_with_token_counts": 0, "requests_with_token_counts": 0,
                        "requests_without_token_counts": 0, "input_tokens": 0, "usd": 0.0, "mean_latency_s": 0.0,
                        "first_day": "mock run", "last_day": "mock run", "sdk_version": exchange.sdk_version(),
                        "model": "mock-jev", "usd_per_million_input_tokens": config.load("jev")["usd_per_mtok_input"]},
                "aws": {**zero_aws, "by_model": {m: dict(zero_aws) for m in COMBINATION}}}
    return {"mode": mode, "jev": jev_usage(), "aws": aws_usage()}


def removed_path(mode: str = "live"):
    return config.results_dir(mode) / "removed_datasets.json"


def removed_share(name: str) -> dict:
    """What was spent on a dataset that is being removed from the study: computed from the request
    log and the record of SageMaker attempts, before its results are deleted. Totals keep it."""
    price = config.load("jev")["usd_per_mtok_input"]
    by_stage: dict[str, dict] = {}
    for p in sorted(exchange.log_dir().glob("*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("dataset") != name:
                    continue
                s = by_stage.setdefault(r["stage"], {"requests": 0, "failed": 0, "input_tokens": 0})
                s["requests"] += 1
                if r.get("status") == 200 and isinstance(r.get("response"), dict):
                    s["input_tokens"] += int(r["response"].get("usage", {}).get("input_tokens", 0))
                else:
                    s["failed"] += 1
    for s in by_stage.values():
        s["usd"] = s["input_tokens"] * price / 1e6
    attempts_path = config.RESULTS / "live" / "baseline_attempts.json"
    attempts = [{"model": "xgb", **a} for a in (json.loads(attempts_path.read_text()) if attempts_path.exists() else [])]
    own = [a for a in attempts if a.get("dataset") == name]
    shared = [a for a in attempts if a.get("dataset") != name and name in str(a.get("dataset", "")).split(",")]
    brief = lambda a: {k: a.get(k) for k in ("model", "kind", "instance_type", "status", "billable_seconds", "cost_usd", "execution_arn")}  # noqa: E731
    return {"dataset": name,
            "jev": {"by_stage": by_stage, "requests": sum(s["requests"] for s in by_stage.values()),
                    "input_tokens": sum(s["input_tokens"] for s in by_stage.values()),
                    "usd": sum(s["usd"] for s in by_stage.values()),
                    "basis": "request-log rows of this dataset; every request made on it was logged"},
            "aws": {"dedicated_executions": [brief(a) for a in own], "dedicated_usd": float(sum(a.get("cost_usd", 0.0) for a in own)),
                    "shared_executions": [brief(a) for a in shared], "shared_usd": float(sum(a.get("cost_usd", 0.0) for a in shared)),
                    "basis": "dedicated: executions whose only dataset was this one; shared: executions that also covered other "
                             "datasets, listed with their whole cost, which cannot be split"},
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def write_removed(name: str, mode: str = "live") -> dict:
    p = removed_path(mode)
    rec = json.loads(p.read_text()) if p.exists() else {"what": "datasets removed from the study after work had been paid for on them; "
                                                               "every total in provenance.json still includes that work",
                                                        "datasets": {}}
    rec["datasets"][name] = removed_share(name)
    p.write_text(json.dumps(rec, indent=1))
    return rec["datasets"][name]


def path(mode: str):
    return config.results_dir(mode) / "provenance.json"


def write(mode: str) -> dict:
    rec = compute(mode)
    path(mode).write_text(json.dumps(rec, indent=1))
    return rec


def read(mode: str) -> dict:
    """The saved usage record. The paper reads this file; it never recomputes usage from the cache."""
    if mode != "live":
        return write(mode)
    if not path(mode).exists():
        raise RuntimeError(f"missing {path(mode)}; run `python -m ciayn provenance --mode live` (needs the response cache "
                           "and the request log)")
    return json.loads(path(mode).read_text())


def run(args) -> int:
    if getattr(args, "removed", None):
        if args.mode != "live":
            raise RuntimeError("--removed records live spending; use --mode live")
        r = write_removed(args.removed)
        print(f"provenance (live): {args.removed}: {r['jev']['requests']} Jev requests, {r['jev']['input_tokens'] / 1e6:.2f}M input "
              f"tokens, ${r['jev']['usd']:.3f}; AWS dedicated ${r['aws']['dedicated_usd']:.2f} "
              f"({len(r['aws']['dedicated_executions'])} executions), shared ${r['aws']['shared_usd']:.2f} "
              f"({len(r['aws']['shared_executions'])} executions); saved to {removed_path()}")
        return 0
    rec = write(args.mode)
    j, a = rec["jev"], rec["aws"]
    print(f"provenance ({args.mode}): {j['requests_in_ledger']} Jev requests in the ledger, {j['requests_in_log']} in the "
          f"request log, {j['requests_without_token_counts']} without a token count; {j['input_tokens'] / 1e6:.1f}M input tokens, "
          f"${j['usd']:.2f}; AWS: {a['executions_succeeded']} executions succeeded of {a['attempts']} attempts, "
          f"{a['billable_seconds']:.0f} billable seconds, ${a['usd']:.2f}")
    return 0
