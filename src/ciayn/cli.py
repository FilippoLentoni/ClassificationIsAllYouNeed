"""Command-line entry point: `python -m ciayn <stage> --mode {mock,live}`."""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

from . import config

# stage -> "module:function"; imported lazily so `--help` stays fast.
STAGES = {
    "freeze": ("ciayn.data.cards:run_freeze", "Freeze the evaluation identifiers and write the dataset documentation"),
    "smoke": ("ciayn.backends.jev:smoke", "One Jev Choice query; prints the full distribution and model version"),
    "template": ("ciayn.templates:run", "Check text templates on training examples and freeze all templates"),
    "predict": ("ciayn.predict:run", "Jev predictions for every dataset (and the mock baseline in mock mode)"),
    "payload-audit": ("ciayn.audit:run", "Report what is known about calls made before request logging"),
    "payload-examples": ("ciayn.payloads:run", "Export one logged request and response per dataset"),
    "baseline": ("ciayn.baseline.cli:run", "XGBoost baseline on SageMaker: plan | deploy | run | fetch | teardown"),
    "metrics": ("ciayn.experiments.metrics:run", "Classification metrics and calibration for every dataset and model"),
    "thresholding": ("ciayn.experiments.thresholding:run", "Threshold selection for binary tasks"),
    "importance": ("ciayn.experiments.importance:run", "Importance-sampling estimation of accuracy"),
    "explain": ("ciayn.experiments.explain:run", "Perturbation explanations and faithfulness on Jev"),
    "comparison": ("ciayn.experiments.comparison:run", "F1 by training labels, cost per prediction and latency for the four models"),
    "escalation": ("ciayn.experiments.escalation:run", "Escalate low-confidence Jev predictions to Qwen (offline)"),
    "escalation-val": ("ciayn.escalation_val:run", "Jev on 500 seeded validation items of AG News (escalation threshold)"),
    "encoder": ("ciayn.encoder.cli:run", "Fine-tuned BERT on SageMaker: plan | run | fetch"),
    "llm": ("ciayn.llm.cli:run", "Self-hosted Qwen3.6-27B on SageMaker: plan | stage | smoke | run | fetch | project-thinking"),
    "remove-dataset": ("ciayn.remove_dataset:run", "Remove a dataset from the study after its spending has been recorded"),
    "provenance": ("ciayn.paper.provenance:run", "Save actual Jev and AWS usage and cost from the ledger, request log and job records"),
    "paper": ("ciayn.paper.build:run", "Generate tables/figures/macros and build the PDF"),
    "verify-paper": ("ciayn.paper.verify:run", "Check every reported number against the saved outputs"),
    "report": ("ciayn.paper.report:run", "Write docs/REPORT.md from the saved outputs"),
    "backup": ("ciayn.backup:run", "Copy the cache, ledger and exchange log outside the working copy"),
}
# live stages that send Jev requests: the cache, ledger and exchange log are backed up after each
JEV_STAGES = ("smoke", "template", "predict", "explain", "escalation-val")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ciayn", description="Classification Is All You Need: experiment pipeline")
    sub = p.add_subparsers(dest="stage", required=True, metavar="{" + "|".join(STAGES) + "}")
    for name, (_, help_) in STAGES.items():
        s = sub.add_parser(name, help=help_)
        if name == "baseline":
            s.add_argument("action", choices=("plan", "deploy", "update", "run", "fetch", "teardown"))
            s.add_argument("--yes", action="store_true", help="deploy/teardown: the user has approved this action")
            s.add_argument("--rerun", action="store_true", help="run: execute again even if a dataset already succeeded")
        if name == "encoder":
            s.add_argument("action", choices=("plan", "run", "fetch"))
            s.add_argument("--rerun", action="store_true", help="run: execute again even if a dataset already succeeded")
        if name == "llm":
            s.add_argument("action", choices=("plan", "stage", "smoke", "run", "fetch", "project-thinking"))
            s.add_argument("--engine", choices=("vllm", "transformers"), default="vllm",
                           help="smoke: which engine's datasets to measure")
            s.add_argument("--thinking", choices=("subsample", "all"), default="subsample",
                           help="run: thinking enabled on the seeded subsample (default) or on every item")
            s.add_argument("--rerun", action="store_true", help="run: execute again even if it already succeeded")
        if name in ("provenance", "remove-dataset"):
            s.add_argument("--removed", metavar="DATASET", help="record what was spent on a dataset that is being removed")
        s.add_argument("--mode", choices=config.MODES, default="mock")
        s.add_argument("--dataset", default="all", choices=("all",) + config.DATASETS)
        s.add_argument("--dry-run", action="store_true", help="report remote calls or jobs that would be made, then stop")
        s.add_argument("--no-build", action="store_true", help="paper: generate files but skip latexmk")
        s.add_argument("--refreeze", action="store_true", help="freeze/template: replace an existing freeze")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)
    target, _ = STAGES[args.stage]
    mod, fn = target.split(":")
    from .backends.budget import DryRunStop

    try:
        rc = getattr(importlib.import_module(mod), fn)(args) or 0
    except DryRunStop:
        return 0
    except (RuntimeError, ValueError) as e:  # user-facing failures (missing key, budget, manifest, ...)
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.stage in JEV_STAGES and args.mode == "live" and not os.environ.get("CIAYN_SKIP_BACKUP"):
        from . import backup

        backup.run()
    return rc


if __name__ == "__main__":
    sys.exit(main())
