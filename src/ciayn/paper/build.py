"""`paper` stage: generated tables, number macros, figures and appendices, then `latexmk` (design D11).

Everything numeric in the paper comes from `paper/generated/`, written here from the saved
outputs under `results/<mode>/` and from `configs/prior_work.yaml`. Every macro remembers the
file it was read from, so `verify-paper` can report each reported number with its source.
The live paper never reads `results/mock/`. If any consumed artifact is mock (or `--mode
mock`), `generated/mockdraft.tex` defines `\\mockdraft`, which switches on the "DRAFT — MOCK
RESULTS" watermark.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import datasets, manifest
from ..experiments.common import MODEL_COLOUR, MODEL_LABEL, plt
from . import provenance

TAG = {"german": "German", "wdbc": "Wdbc", "sst2": "Sst", "agnews": "Agnews"}
MODEL_TAG = {"jev": "Jev", "qwen": "Qwen", "xgb": "Xgb", "bert": "Bert", "qwen_think": "QwenThink"}
ROW_LABEL = {"jev": "Jev, zero-shot", "qwen": "Qwen3.6-27B, zero-shot", "xgb": "XGBoost, supervised",
             "bert": "BERT, supervised", "qwen_think": "Qwen3.6-27B thinking, zero-shot"}
SHORT = {"jev": "Jev", "qwen": "Qwen3.6-27B", "xgb": "XGBoost", "bert": "BERT", "qwen_think": "Qwen3.6-27B, thinking"}
DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
          "8": "Eight", "9": "Nine"}


def macro_name(*parts) -> str:
    """LaTeX macro names are letters only: 'macro_f1' -> 'MacroFOne'."""
    words = re.split(r"[^A-Za-z0-9]+", "_".join(str(p) for p in parts))
    name = "".join(w[:1].upper() + w[1:] for w in words if w)
    return "".join(DIGITS.get(ch, ch) for ch in name)


def tex_escape(s) -> str:
    s = str(s).replace("\\", r"\textbackslash{}")
    s = re.sub(r"([_%&#$\{\}])", r"\\\1", s)
    return s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


class Raw(str):
    """Text that is already valid LaTeX and must not be escaped."""


def breakable(tex: str) -> str:
    """Let long hashes, paths and timestamps break across lines."""
    tex = re.sub(r"([0-9a-f]{8})(?=[0-9a-f]{4})", r"\1\\allowbreak{}", tex)
    return re.sub(r"(?<=[A-Za-z0-9])([/:])(?=[A-Za-z0-9])", r"\1\\allowbreak{}", tex)


def thousands(n) -> Raw:
    return Raw(f"{int(n):,}".replace(",", "{,}"))


def short(title: str) -> str:
    return title if len(title) < 22 else "Breast Cancer Wisconsin"


class Generated:
    def __init__(self, mode: str, out: Path | None = None):
        self.mode = mode
        self.out = out or (config.PAPER / "generated")
        self.macros: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        self.artifact_modes: set[str] = set()
        self.src = ""

    def _rel(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(config.ROOT))
        except ValueError:
            return str(path)

    def read(self, dataset: str, stage: str, name: str) -> dict:
        path = config.RESULTS / self.mode / dataset / stage / f"{name}.json"
        if not path.exists():
            raise RuntimeError(f"missing {path}; run `python -m ciayn {stage} --mode {self.mode}` first")
        d = json.loads(path.read_text())
        self.artifact_modes.update(d.get("artifact_modes", [d.get("mode", self.mode)]))
        self.src = self._rel(path)
        return d

    def use(self, path) -> None:
        self.src = self._rel(path)

    def macro(self, value, *name, fmt: str = "{:.3f}") -> str:
        if isinstance(value, Raw):
            text = str(value)
        elif isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            text = tex_escape(value)
        else:
            text = fmt.format(value)
        key = macro_name(*name)
        if key in self.macros and self.macros[key] != text:
            raise RuntimeError(f"macro \\{key} defined twice with different values")
        self.macros[key], self.sources[key] = text, self.src
        return text

    def table(self, name: str, tex: str) -> None:
        (self.out / f"{name}.tex").write_text(tex)

    def savefig(self, fig, name: str) -> None:
        fig.tight_layout()
        if self.mode == "mock":
            fig.text(0.5, 0.5, "MOCK RESULTS", fontsize=30, color="red", alpha=0.25, ha="center", va="center",
                     rotation=20, transform=fig.transFigure)
        fig.savefig(self.out / "figs" / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)


def unc(h: dict, digits: int = 3) -> str:
    """A value with its uncertainty: a bootstrap interval, or the standard deviation over repeats."""
    v = f"{h['value']:.{digits}f}"
    if "ci95" in h:
        return f"{v}\\,{{\\scriptsize[{h['ci95'][0]:.{digits}f},\\,{h['ci95'][1]:.{digits}f}]}}"
    return f"{v}\\,{{\\scriptsize$\\pm${h['sd']:.{digits}f}}}"


def booktabs(spec: str, header: str, rows: list[str]) -> str:
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


# ---- provenance and datasets -------------------------------------------------------------------

def provenance_part(g: Generated) -> dict:
    p = provenance.read(g.mode)
    g.use(config.RESULTS / g.mode / "provenance.json")
    j, a = p["jev"], p["aws"]["by_model"]["xgb"]          # the XGBoost macros describe the XGBoost runs only
    for m, u in p["aws"]["by_model"].items():
        g.macro(u["usd"], "AwsCostUSD", MODEL_TAG[m], fmt="{:.2f}")
        g.macro(u["billable_seconds"] / 60, "AwsBillableMinutes", MODEL_TAG[m], fmt="{:.0f}")
        g.macro(u["gpu_billable_seconds"] / 3600, "AwsGpuHours", MODEL_TAG[m], fmt="{:.2f}")
        g.macro(u["executions_started"], "AwsExecutions", MODEL_TAG[m], fmt="{:d}")
        g.macro(u["executions_succeeded"], "AwsExecutionsSucceeded", MODEL_TAG[m], fmt="{:d}")
        g.macro(u["instance_type"] or "not run", "AwsInstanceType", MODEL_TAG[m])
    g.macro(p["aws"]["usd"], "AwsCostTotalUSD", fmt="{:.2f}")
    g.macro(p["aws"]["billable_seconds"] / 3600, "AwsBillableHoursTotal", fmt="{:.1f}")
    g.macro(j["model"], "JevModel")
    g.macro(j["sdk_version"], "SdkVersion")
    g.macro(thousands(j["cap"]), "JevCap")
    g.macro(thousands(j["requests_in_ledger"]), "JevInferencesUsed")
    for stage, s in j["by_stage"].items():
        g.macro(thousands(s["requests"]), "JevRequests", stage)
    g.macro(thousands(j["requests_in_log"]), "JevRequestsLogged")
    g.macro(thousands(j["requests_before_logging"]), "JevRequestsBeforeLogging")
    g.macro(thousands(j["requests_before_logging_with_token_counts"]), "JevRequestsBeforeLoggingCached")
    g.macro(thousands(j["requests_with_token_counts"]), "JevRequestsWithTokens")
    g.macro(thousands(j["requests_without_token_counts"]), "JevRequestsWithoutTokens")
    g.macro(thousands(j["failed_requests"]), "JevFailedRequests")
    g.macro(j["input_tokens"] / 1e6, "JevInputTokensM", fmt="{:.1f}")
    g.macro(j["usd"], "JevCostUSD", fmt="{:.2f}")
    g.macro(j["usd_per_million_input_tokens"], "JevPricePerMTok", fmt="{:.3f}")
    g.macro(j["mean_latency_s"] or 0.0, "JevLatency", fmt="{:.2f}")
    g.macro(j["first_day"] if j["first_day"] == j["last_day"] else f"{j['first_day']} to {j['last_day']}", "RunDate")
    g.macro(a["executions_started"], "AwsExecutions", fmt="{:d}")
    g.macro(a["executions_succeeded"], "AwsExecutionsSucceeded", fmt="{:d}")
    g.macro(a["executions_not_completed"], "AwsExecutionsNotCompleted", fmt="{:d}")
    g.macro(a["executions_superseded"], "AwsExecutionsSuperseded", fmt="{:d}")
    g.macro(len(a["datasets"]), "AwsDatasetsDone", fmt="{:d}")
    g.macro(a["jobs"], "AwsJobs", fmt="{:d}")
    g.macro(a["attempts"], "AwsAttempts", fmt="{:d}")
    g.macro(a["refused_attempts"], "AwsRefusedAttempts", fmt="{:d}")
    g.macro(a["billable_seconds"] / 60, "AwsBillableMinutes", fmt="{:.0f}")
    g.macro(a["usd"], "AwsCostUSD", fmt="{:.2f}")
    g.macro(a["spend_limit_usd"], "AwsSpendLimit", fmt="{:.0f}")
    g.macro(a["instance_type"], "AwsInstanceType")
    g.macro(a["region"], "AwsRegion")
    images = {i.split("/")[-1].split(":")[0]: i.split(":")[-1] for i in a["images"]}
    g.macro(images.get("sagemaker-xgboost", "none (mock)"), "XgbImageTag")
    g.macro(images.get("sagemaker-scikit-learn", "none (mock)"), "SklearnImageTag")
    cfg = config.load("baseline")
    g.use(config.CONFIGS / "baseline.yaml")
    g.macro(cfg["max_rounds"], "XgbMaxRounds", fmt="{:d}")
    g.macro(cfg["patience"], "XgbPatience", fmt="{:d}")
    g.macro(thousands(cfg["text"]["max_features"]), "TfidfFeatures")
    g.macro(cfg["text"]["max_bin"], "XgbTextBins", fmt="{:d}")
    g.macro(len(cfg["grid"]["max_depth"]) * len(cfg["grid"]["eta"]), "XgbGridSize", fmt="{:d}")
    g.use(config.RESULTS / g.mode / "provenance.json")
    rows = [f"{tex_escape(stage)} & {thousands(s['requests'])} & {thousands(s['logged'])} & {s['input_tokens'] / 1e6:.2f} & {s['usd']:.3f} \\\\"
            for stage, s in j["by_stage"].items()]
    rows.append(f"\\midrule\nall stages & {thousands(j['requests_in_ledger'])} & {thousands(j['requests_in_log'])} & "
                f"{j['input_tokens'] / 1e6:.2f} & {j['usd']:.3f} \\\\")
    removed = config.RESULTS / g.mode / "removed_datasets.json"
    names = ("RemovedDataset", "RemovedJevRequests", "RemovedJevTokensM", "RemovedJevUSD", "RemovedAwsDedicatedUSD",
             "RemovedAwsSharedUSD", "RemovedAwsDedicatedExecutions", "RemovedAwsSharedExecutions")
    if removed.exists():
        rec = json.loads(removed.read_text())["datasets"]
        if len(rec) != 1:
            raise RuntimeError("the paper states the share of exactly one removed dataset")
        (name, r), = rec.items()
        g.use(removed)
        g.macro({"banking77": "Banking77"}.get(name, name), names[0])
        g.macro(thousands(r["jev"]["requests"]), names[1])
        g.macro(r["jev"]["input_tokens"] / 1e6, names[2], fmt="{:.1f}")
        g.macro(r["jev"]["usd"], names[3], fmt="{:.2f}")
        g.macro(r["aws"]["dedicated_usd"], names[4], fmt="{:.2f}")
        g.macro(r["aws"]["shared_usd"], names[5], fmt="{:.2f}")
        g.macro(len(r["aws"]["dedicated_executions"]), names[6], fmt="{:d}")
        g.macro(len(r["aws"]["shared_executions"]), names[7], fmt="{:d}")
        rows.append(f"of which {tex_escape(g.macros[names[0]])} (removed) & {thousands(r['jev']['requests'])} & "
                    f"{thousands(r['jev']['requests'])} & {r['jev']['input_tokens'] / 1e6:.2f} & {r['jev']['usd']:.3f} \\\\")
    else:
        for name in names:
            g.macros[name], g.sources[name] = "none", "no dataset was removed in these results"
    g.macro(len(config.DATASETS), "NDatasets", fmt="{:d}")
    g.table("tab_cost", booktabs("lrrrr", "Stage & Requests & In request log & Input tokens (M) & Cost (USD)", rows))
    return p


def datasets_part(g: Generated) -> dict:
    dss = {n: datasets.load(n) for n in config.DATASETS}
    cfg = config.load("datasets")
    g.use(config.SPLITS / manifest.MANIFEST)
    g.macro(manifest.frozen_at(), "FrozenAt")
    g.macro(cfg["repeats"], "NRepeats", fmt="{:d}")
    g.use(config.CONFIGS / "datasets.yaml")
    fr = cfg["repeat_fractions"]
    g.macro(Raw(f"{round(100 * (1 - fr['val'] - fr['test']))}/{round(100 * fr['val'])}/{round(100 * fr['test'])}"), "SplitFractions")
    from ..templates import N_CHECK
    g.macro(N_CHECK, "TemplateCheckN", fmt="{:d}")
    rows = []
    for n, ds in dss.items():
        g.use(config.SPLITS / f"{n}.json")
        r0 = ds.repeats[0]
        g.macro(thousands(len(ds.jev_ids)), "NJev", TAG[n])
        g.macro(thousands(len(r0["test"])), "NTest", TAG[n])
        g.macro(thousands(len(r0["val"])), "NVal", TAG[n])
        g.macro(thousands(len(r0["train"])), "NTrain", TAG[n])
        g.macro(len(ds.classes), "NClasses", TAG[n], fmt="{:d}")
        if ds.protocol == "repeated":
            proto = f"{len(ds.repeats)} stratified repeats, test {len(r0['test'])} each"
            tv = f"{len(r0['train'])} / {len(r0['val'])} per repeat"
        else:
            proto = f"official {cfg[n]['eval_split']} split, {thousands(len(r0['test']))}"
            tv = f"{thousands(len(r0['train']))} / {thousands(len(r0['val']))}"
        rows.append(f"{tex_escape(ds.title)} & {ds.kind} & {len(ds.classes)} & {proto} & {tv} & {thousands(len(ds.jev_ids))} \\\\")
    g.table("tab_datasets", booktabs("llrllr", "Dataset & Type & Classes & Evaluation examples & Train / validation & Jev requests", rows))
    small = []
    for n, ds in dss.items():
        r0 = ds.repeats[0]
        ev = f"{len(ds.repeats)} $\\times$ {len(r0['test'])}" if ds.protocol == "repeated" else f"{thousands(len(r0['test']))}"
        small.append(f"{tex_escape(short(ds.title))} & {ds.kind} & {len(ds.classes)} & {ev} & {thousands(len(r0['train']))} & "
                     f"{thousands(len(r0['val']))} \\\\")
    g.table("tab_datasets_main", booktabs("llrrrr", "Dataset & Type & Classes & Evaluation & Training & Validation", small))
    for n, ds in dss.items():                     # class balance of the evaluation examples (first repeat; all repeats are stratified alike)
        g.use(config.SPLITS / f"{n}.json")
        counts = np.bincount(ds.labels(ds.ids("test", 0)), minlength=len(ds.classes))
        if ds.positive:
            g.macro(thousands(counts[1]), "NTestPositive", TAG[n])
            g.macro(thousands(counts[0]), "NTestNegative", TAG[n])
            g.macro(100 * counts[1] / counts.sum(), "PctTestPositive", TAG[n], fmt="{:.0f}")
        else:
            g.macro(thousands(counts.min()), "NTestPerClassMin", TAG[n])
            g.macro(thousands(counts.max()), "NTestPerClassMax", TAG[n])
    g.use(config.SPLITS / "german.json")
    g.macro(len(dss["german"].eval_ids), "NDistinctTest", "German", fmt="{:d}")
    g.use(config.SPLITS / "wdbc.json")
    g.macro(len(dss["wdbc"].eval_ids), "NDistinctTest", "Wdbc", fmt="{:d}")
    g.use(config.SPLITS / "sst2.json")
    g.macro(len(dss["sst2"].jev_val), "NJevVal", "Sst", fmt="{:d}")
    return dss


# ---- prior public evaluations ------------------------------------------------------------------

def prior_part(g: Generated, metrics: dict) -> None:
    path = config.CONFIGS / "prior_work.yaml"
    prior = config.load("prior_work")
    rows, seen = [], set()
    for src in prior["sources"]:
        if not src.get("checked_on"):
            raise RuntimeError(f"prior_work.yaml: source {src.get('key')} has no check date; a prior number may be "
                               "cited only after it was checked against its source")
        for r in src["results"]:
            for field in ("dataset", "n", "accuracy", "quote"):
                if field not in r:
                    raise RuntimeError(f"prior_work.yaml: a result of {src['key']} lacks `{field}`")
            ds = r["dataset"]
            seen.add(ds)
            if ds not in metrics:
                continue
            g.use(path)
            g.macro(r["accuracy"], "PriorAcc", src["tag"], TAG[ds])
            if "ece" in r:
                g.macro(r["ece"], "PriorEce", src["tag"], TAG[ds])
            g.macro(thousands(r["n"]), "PriorN", src["tag"], TAG[ds])
            ours = metrics[ds]["models"]["jev"]
            ece = f"{r['ece']:.3f}" if "ece" in r else "--"
            rows.append(f"{tex_escape(datasets.load(ds).title)} & {tex_escape(src['label'])}~\\citep{{{src['key']}}} & "
                        f"{thousands(r['n'])} & {r['accuracy']:.3f} & {ece} & {thousands(ours['n_evaluated'])} & "
                        f"{ours['headline']['accuracy']['value']:.3f} & {ours['headline']['ece']['value']:.3f} \\\\")
    for ds in prior.get("none_found", []):
        seen.add(ds)
        if ds not in metrics:
            continue
        ours = metrics[ds]["models"]["jev"]
        rows.append(f"{tex_escape(datasets.load(ds).title)} & none found & -- & -- & -- & {thousands(ours['n_evaluated'])} & "
                    f"{ours['headline']['accuracy']['value']:.3f} & {ours['headline']['ece']['value']:.3f} \\\\")
        seen.add(ds)
    missing = set(config.DATASETS) - seen
    if missing:
        raise RuntimeError(f"prior_work.yaml says nothing about {sorted(missing)}: list a checked source or add them to none_found")
    g.use(path)
    g.macro(str(max(s["checked_on"] for s in prior["sources"])), "PriorCheckedOn")
    g.table("tab_prior", "\\begin{tabular}{llrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{Prior public evaluation} & "
            "\\multicolumn{3}{c}{This paper (Jev, zero-shot)} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Dataset & Source & $n$ & Acc. & ECE & $n$ & Acc. & ECE \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")


# ---- metrics and calibration -------------------------------------------------------------------

NA = "\\multicolumn{4}{c}{\\emph{not available: the run had not completed}}"


def na(model: str, dataset: str) -> str:
    """A model without a result on a dataset is shown as not available, never filled in."""
    return NA


def metrics_part(g: Generated) -> dict:
    res = {n: g.read(n, "metrics", "metrics") for n in config.DATASETS}
    main, calib, labels = [], [], []
    for n, m in res.items():
        g.use(config.RESULTS / g.mode / n / "metrics" / "metrics.json")
        title = tex_escape(short(m["title"]))
        for i, model in enumerate(config.MODELS):
            first = f"\\multirow{{{len(config.MODELS)}}}{{*}}{{{title}}}" if i == 0 else ""
            if model not in m["models"]:        # never filled in: the row says so
                for rows in (main, calib):
                    rows.append(f"{first} & {ROW_LABEL[model]} & {na(model, n)} \\\\")
                labels.append(f"{first} & {ROW_LABEL[model]} & {na(model, n)} & \\\\")
                continue
            r = m["models"][model]
            h = r["headline"]
            main.append(f"{first} & {ROW_LABEL[model]} & {unc(h['accuracy'])} & {unc(h['macro_f1'])} & {unc(h['roc_auc'])} & {unc(h['pr_auc'])} \\\\")
            calib.append(f"{first} & {ROW_LABEL[model]} & {unc(h['ece'])} & {unc(h['brier'])} & {unc(h['log_loss'])} & {r['mean_confidence']:.3f} \\\\")
            lb = m["label_budget"][model]
            esc_path = config.RESULTS / g.mode / n / "escalation" / "escalation.json"
            esc = (thousands(json.loads(esc_path.read_text())["repeats"][0]["n_validation_labels"])
                   if model in config.ZERO_SHOT and esc_path.exists() else "--")
            labels.append(f"{first} & {ROW_LABEL[model]} & {thousands(lb['training'])} & {thousands(lb['early_stopping_and_selection'])} & "
                          f"{thousands(lb['calibration'])} & {thousands(lb['threshold_selection'])} & {esc} \\\\")
            for key in ("accuracy", "macro_f1", "roc_auc", "pr_auc", "ece", "brier", "log_loss"):
                g.macro(h[key]["value"], key, MODEL_TAG[model], TAG[n])
            g.macro(100 * h["accuracy"]["value"], "AccPct", MODEL_TAG[model], TAG[n], fmt="{:.1f}")
            g.macro(r["mean_confidence"], "MeanConf", MODEL_TAG[model], TAG[n])
            g.macro(r["invalid_rate"], "InvalidRate", MODEL_TAG[model], TAG[n])
            if m["positive"]:
                g.macro(max(r["true_share"]), "MajorityRate", TAG[n])
                g.macro(r["predicted_share"][1], "PredictedPositive", MODEL_TAG[model], TAG[n], fmt="{:.2f}")
                g.macro(r["true_share"][1], "TruePositive", TAG[n], fmt="{:.2f}")
                g.macro(r["per_class"][1]["recall"], "RecallPositive", MODEL_TAG[model], TAG[n])
                g.macro(r["per_class"][0]["recall"], "RecallNegative", MODEL_TAG[model], TAG[n])
        if n != config.DATASETS[-1]:
            main.append("\\addlinespace"), calib.append("\\addlinespace"), labels.append("\\addlinespace")
        for model, curve in m["learning_curves"].items():
            suffix = () if model == "xgb" else (MODEL_TAG[model],)
            g.macro(thousands(curve[0]["n_train"]), "CurveSmallestN", *suffix, TAG[n])
            g.macro(curve[0]["accuracy"]["value"], "CurveSmallestAcc", *suffix, TAG[n])
    g.src = f"results/{g.mode}/*/jev/predictions.json"
    summaries = [json.loads((config.RESULTS / g.mode / n / "jev" / "predictions.json").read_text()) for n in res]
    g.macro(sum(s["invalid"] for s in summaries), "JevInvalidOutputs", fmt="{:d}")
    g.macro(thousands(sum(s["n_rows"] for s in summaries)), "JevPredictions")
    g.use(config.RESULTS / g.mode / "german" / "metrics" / "metrics.json")
    g.macro(res["german"]["ece_bins"], "ECEBins", fmt="{:d}")
    g.use(config.CONFIGS / "metrics.yaml")
    g.macro(thousands(config.load("metrics")["bootstrap"]), "BootstrapN")
    from ..experiments.metrics import LOG_LOSS_CLIP
    g.use("src/ciayn/experiments/metrics.py")
    g.macro(Raw(f"$10^{{{int(np.log10(LOG_LOSS_CLIP))}}}$"), "LogLossClip")
    compact = []
    for n, m in res.items():
        for i, model in enumerate(config.MODELS):
            first = f"\\multirow{{{len(config.MODELS)}}}{{*}}{{{tex_escape(short(m['title']))}}}" if i == 0 else ""
            if model not in m["models"]:
                compact.append(f"{first} & {SHORT[model]} & \\multicolumn{{4}}{{c}}{{\\emph{{not available}}}} \\\\")
                continue
            h = m["models"][model]["headline"]
            compact.append(f"{first} & {SHORT[model]} & {unc(h['accuracy'])} & {h['macro_f1']['value']:.3f} & "
                           f"{h['roc_auc']['value']:.3f} & {h['ece']['value']:.3f} \\\\")
        if n != config.DATASETS[-1]:
            compact.append("\\addlinespace")
    g.table("tab_main", booktabs("llcccc", "Dataset & Model & Accuracy & Macro-F1 & ROC-AUC & ECE", compact))
    g.table("tab_metrics", booktabs("llcccc", "Dataset & Model & Accuracy & Macro-F1 & ROC-AUC & PR-AUC", main))
    g.table("tab_calibration", booktabs("llcccc", "Dataset & Model & ECE & Brier & Log-loss & Mean confidence", calib))
    g.table("tab_labels", booktabs("llrrrrr", "Dataset & Model & Training & Early stopping, selection & Calibration & Threshold selection & "
                                   "Escalation threshold", labels))
    baseline_text(g, res)

    curve_rows = []
    for n, m in res.items():
        zero = "; ".join(f"{ROW_LABEL[mo].split(',')[0]} {m['models'][mo]['headline']['accuracy']['value']:.3f} / "
                         f"{m['models'][mo]['headline']['macro_f1']['value']:.3f}" for mo in config.ZERO_SHOT if mo in m["models"])
        rows = [f"0 & \\multicolumn{{4}}{{l}}{{{zero}}}"]
        sizes = sorted({c["n_train"] for mo in config.SUPERVISED for c in m["learning_curves"].get(mo, [])})
        for size in sizes:
            cells = []
            for mo in config.SUPERVISED:
                c = next((c for c in m["learning_curves"].get(mo, []) if c["n_train"] == size), None)
                cells.append(f"{c['accuracy']['value']:.3f} & {c['macro_f1']['value']:.3f}" if c else "-- & --")
            rows.append(f"{thousands(size)} & " + " & ".join(cells))
        title = f"\\multirow{{{len(rows)}}}{{*}}{{{tex_escape(short(m['title']))}}}"
        curve_rows += [f"{title if i == 0 else ''} & {r} \\\\" for i, r in enumerate(rows)]
        if n != config.DATASETS[-1]:
            curve_rows.append("\\addlinespace")
    g.table("tab_learning", "\\begin{tabular}{lrcccc}\n\\toprule\n & & " +
            " & ".join(f"\\multicolumn{{2}}{{c}}{{{ROW_LABEL[mo]}}}" for mo in config.SUPERVISED) +
            " \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nDataset & Training labels & Accuracy & Macro-F1 & Accuracy & Macro-F1 \\\\\n"
            "\\midrule\n" + "\n".join(curve_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    nm = len(config.MODELS)
    fig, axes = plt.subplots(nm, len(res), figsize=(2.2 * len(res), 2.1 * nm + 0.1), sharex=True, sharey=True)
    for col, (n, m) in enumerate(res.items()):
        for row, model in enumerate(config.MODELS):
            ax = axes[row, col]
            ax.set(xlim=(0, 1), ylim=(0, 1))
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\naccuracy", fontsize=8)
            if row == nm - 1:
                ax.set_xlabel("confidence", fontsize=8)
            if model not in m["models"]:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=8, color="grey")
                continue
            rel = m["models"][model]["reliability"]
            ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
            ax.bar([(b["lo"] + b["hi"]) / 2 for b in rel], [b["accuracy"] for b in rel], width=1 / m["ece_bins"], alpha=0.75,
                   color=MODEL_COLOUR[model], edgecolor="k", lw=0.3)
            ece = f"ECE {m['models'][model]['headline']['ece']['value']:.3f}"
            ax.set_title(f"{short(m['title'])}\n{ece}" if row == 0 else ece, fontsize=8)
    g.savefig(fig, "reliability")

    fig, axes = plt.subplots(1, len(res), figsize=(2.2 * len(res), 2.5))
    short_label = {"jev": "Jev", "qwen": "Qwen3.6-27B", "xgb": "XGBoost", "bert": "BERT"}
    for ax, (n, m) in zip(axes, res.items()):
        for model, c in m["learning_curves"].items():
            ax.errorbar([r["n_train"] for r in c], [r["accuracy"]["value"] for r in c], yerr=[r["accuracy"]["sd"] for r in c],
                        marker="o", ms=3, c=MODEL_COLOUR[model], capsize=2, lw=1, label=short_label[model])
        c = next(iter(m["learning_curves"].values()), None)
        if c:
            ax.set_xscale("log")
            ax.set_xticks([c[0]["n_train"], c[len(c) // 2]["n_train"], c[-1]["n_train"]],
                          [f"{r['n_train']:,}" for r in (c[0], c[len(c) // 2], c[-1])])
            ax.xaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.text(0.5, 0.35, "supervised models\nnot available", ha="center", va="center", fontsize=8, color="grey", transform=ax.transAxes)
            ax.set_xticks([])
        for model in config.ZERO_SHOT:
            if model in m["models"]:
                ax.axhline(m["models"][model]["headline"]["accuracy"]["value"], c=MODEL_COLOUR[model], ls="--", lw=1,
                           label=f"{short_label[model]}, zero-shot")
        ax.set_title(short(m["title"]), fontsize=8)
        ax.set_xlabel("training labels", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("accuracy", fontsize=8)
    axes[0].legend(fontsize=7)
    g.savefig(fig, "learning_curves")
    return res


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def baseline_text(g: Generated, res: dict) -> None:
    """The sentences about the baseline on the text datasets, written from whichever baseline runs exist."""
    text = [n for n in res if datasets.load(n).kind == "text"]
    have = [n for n in text if "xgb" in res[n]["models"]]
    miss = [n for n in text if n not in have]
    name = lambda n: tex_escape(short(res[n]["title"]))  # noqa: E731
    out = []
    if have:
        out.append("On the text datasets, XGBoost sees only TF-IDF features and reaches " +
                   _join([f"{res[n]['models']['xgb']['headline']['accuracy']['value']:.3f} on {name(n)}" for n in have]) +
                   " (zero-shot Jev: " + _join([f"{res[n]['models']['jev']['headline']['accuracy']['value']:.3f}" for n in have]) + ").")
        parts = []
        for n in have:
            jev = res[n]["models"]["jev"]["headline"]["accuracy"]["value"]
            curve = res[n]["learning_curves"]["xgb"]
            reach = next((c["n_train"] for c in curve if c["accuracy"]["value"] >= jev), None)
            if reach is None:
                parts.append(f"on {name(n)} it does not reach zero-shot Jev even with all {thousands(curve[-1]['n_train'])} training labels")
            elif reach == curve[0]["n_train"]:
                parts.append(f"on {name(n)} it is already above zero-shot Jev at the smallest size tried, {thousands(reach)} labels")
            else:
                parts.append(f"on {name(n)} it first reaches zero-shot Jev at {thousands(reach)} training labels")
        out.append("The learning curves (Figure~\\ref{fig:learning}) put the comparison in terms of labels: " +
                   "; ".join(parts) + ".")
    if miss:
        one = len(miss) == 1
        out.append("\\textbf{The XGBoost " + ("run" if one else "runs") + " for " + _join([name(n) for n in miss]) +
                   " had not completed when this version was built; " + ("its baseline row is" if one else "their baseline rows are") +
                   " marked as not available and nothing is reported for " + ("it" if one else "them") + ".}")
    g.table("text_baseline", "\n".join(out) + "\n")
    g.use(f"results/{g.mode}/*/metrics/metrics.json")
    g.macro(len(miss), "BaselinesMissing", fmt="{:d}")


# ---- thresholds --------------------------------------------------------------------------------

def thresholding_part(g: Generated) -> None:
    label = {"default": "default", "analytic": "analytic", "validation": "validation-selected"}

    def cell(h: dict) -> str:
        return f"{h['value']:.3f}" + (f"\\,{{\\scriptsize$\\pm${h['sd']:.3f}}}" if h["n_repeats"] > 1 else "")

    rows = []
    for n in config.BINARY:
        t = g.read(n, "thresholding", "thresholding")
        for mi, model in enumerate(mo for mo in ("jev", "xgb") if mo in t["models"]):
            sm = t["models"][model]["summary"]
            at = sm["at"]
            first = f"\\multirow{{2}}{{*}}{{{tex_escape(short(t['title']))}}}" if mi == 0 else ""
            rows.append(f"{first} & {SHORT[model]} & {at['default']['mean_cost']['value']:.3f} & {at['analytic']['mean_cost']['value']:.3f} & "
                        f"{sm['thresholds']['validation']['value']:.2f} & {at['validation']['mean_cost']['value']:.3f} & "
                        f"{at['default']['accuracy']['value']:.3f} & {at['validation']['accuracy']['value']:.3f} \\\\")
        if n != config.BINARY[-1]:
            rows.append("\\addlinespace")
    g.table("tab_thresholds_main", "\\begin{tabular}{llcccccc}\n\\toprule\n & & \\multicolumn{2}{c}{Cost at fixed $t$} & "
            "\\multicolumn{2}{c}{Validation-selected} & \\multicolumn{2}{c}{Accuracy} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Dataset & Model & default & analytic & $t$ & cost & default & selected \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    # confusion counts for the imbalanced tabular datasets, pooled over the split repeats
    rows, tabular = [], [n for n in config.BINARY if datasets.load(n).kind == "tabular"]
    for n in tabular:
        t = g.read(n, "thresholding", "thresholding")
        cm = t["cost_matrix"]["rows_true_cols_pred"]
        present = [mo for mo in config.MODELS if mo in t["models"]]
        for mi, model in enumerate(present):
            sm = t["models"][model]["summary"]
            cells = []
            for point in ("default", "validation"):
                c = sm["counts"][point]
                total = cm[1][0] * c["fn"] + cm[0][1] * c["fp"]
                cells.append(f"{thousands(c['tp'])} & {thousands(c['fn'])} & {thousands(c['fp'])} & {thousands(c['tn'])} & {thousands(total)}")
                for k in ("tp", "fn", "fp", "tn"):
                    g.macro(thousands(c[k]), "Conf", k, point, MODEL_TAG[model], TAG[n])
                g.macro(thousands(total), "ConfCost", point, MODEL_TAG[model], TAG[n])
            first = f"\\multirow{{{len(present) + 1}}}{{*}}{{{tex_escape(short(t['title']))}}}" if mi == 0 else ""
            rows.append(f"{first} & {SHORT[model]} & " + " & ".join(cells) + " \\\\")
        c = t["models"]["jev"]["summary"]["counts"]["default"]
        pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
        g.macro(thousands(pos), "ConfPositives", TAG[n])
        g.macro(thousands(neg), "ConfNegatives", TAG[n])
        g.macro(thousands(cm[0][1] * neg), "ConfCostFlagAll", TAG[n])
        g.macro(thousands(cm[1][0] * pos), "ConfCostFlagNone", TAG[n])
        rows.append(f" & \\emph{{flag every case}} & {thousands(pos)} & 0 & {thousands(neg)} & 0 & {thousands(cm[0][1] * neg)} & "
                    f"{thousands(pos)} & 0 & {thousands(neg)} & 0 & {thousands(cm[0][1] * neg)} \\\\")
        if n != tabular[-1]:
            rows.append("\\addlinespace")
    g.table("tab_confusion", "\\begin{tabular}{llrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{5}{c}{Default decision} & "
            "\\multicolumn{5}{c}{Validation-selected threshold} \\\\\n\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
            "Dataset & Model & TP & FN & FP & TN & Cost & TP & FN & FP & TN & Cost \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    header = "Dataset & Model & Threshold & $t$ & Precision & Recall & F1 (pos.) & Accuracy & Mean cost"
    for name_, keep in (("tab_thresholds", ("jev", "xgb")), ("tab_thresholds_all", config.MODELS)):
        g.table(name_, booktabs("lllcccccc", header, _threshold_rows(g, keep, label, cell)))


def _threshold_rows(g: Generated, keep: tuple, label: dict, cell) -> list[str]:
    """Rows for the models in `keep` (the main-text table shows Jev and XGBoost; the appendix shows all)."""
    rows = []
    for n in config.BINARY:
        t = g.read(n, "thresholding", "thresholding")
        present = [mo for mo in keep if mo in t["models"]]
        for mi, model in enumerate(present):
            s = t["models"][model]["summary"]
            for ti, name in enumerate(("default", "analytic", "validation")):
                a = s["at"][name]
                first = f"\\multirow{{{3 * len(present)}}}{{*}}{{{tex_escape(short(t['title']))}}}" if mi == 0 and ti == 0 else ""
                second = f"\\multirow{{3}}{{*}}{{{ROW_LABEL[model]}}}" if ti == 0 else ""
                rows.append(f"{first} & {second} & {label[name]} & {s['thresholds'][name]['value']:.2f} & {cell(a['precision'])} & "
                            f"{cell(a['recall'])} & {cell(a['f1_positive'])} & {cell(a['accuracy'])} & {cell(a['mean_cost'])} \\\\")
                for key in ("precision", "recall", "f1_positive", "accuracy", "mean_cost"):
                    g.macro(a[key]["value"], "Thr", key, name, MODEL_TAG[model], TAG[n])
                g.macro(s["thresholds"][name]["value"], "Threshold", name, MODEL_TAG[model], TAG[n], fmt="{:.2f}")
            g.macro(t["models"][model]["n_validation_labels"], "ThrValLabels", MODEL_TAG[model], TAG[n], fmt="{:d}")
            if mi < len(present) - 1:
                rows.append("\\cmidrule(lr){2-9}")
        c = t["cost_matrix"]["rows_true_cols_pred"]
        g.macro(f"{c[1][0]:g}:{c[0][1]:g}", "CostRatio", TAG[n])
        g.macro("illustrative" if t["cost_matrix"]["illustrative"] else "canonical", "CostKind", TAG[n])
        if n != config.BINARY[-1]:
            rows.append("\\midrule")
        for model in present:
            shutil.copyfile(config.RESULTS / g.mode / n / "thresholding" / f"threshold_sweep_{model}.pdf",
                            g.out / "figs" / f"threshold_{n}_{model}.pdf")
    return rows


# ---- importance sampling -----------------------------------------------------------------------

def importance_part(g: Generated) -> None:
    res = {n: g.read(n, "importance", "importance") for n in config.DATASETS}
    shown = [b for b in (25, 100, 400) if all(str(b) in r["samplers"]["uniform"] for r in res.values())]
    rows = []
    for n, r in res.items():
        g.use(config.RESULTS / g.mode / n / "importance" / "importance.json")
        cells = []
        for b in shown:
            u, s, ratio = r["samplers"]["uniform"][str(b)], r["samplers"]["importance"][str(b)], r["variance_ratio_uniform_over_importance"][str(b)]
            cells.append(f"{u['rmse']:.3f} & {s['rmse']:.3f} & {ratio:.2f}")
            g.macro(u["rmse"], "IsRmseUniform", TAG[n], "n", b)
            g.macro(s["rmse"], "IsRmseGuided", TAG[n], "n", b)
            g.macro(ratio, "IsVarRatio", TAG[n], "n", b, fmt="{:.2f}")
            g.macro(s["coverage95"], "IsCoverageGuided", TAG[n], "n", b, fmt="{:.2f}")
            g.macro(u["coverage95"], "IsCoverageUniform", TAG[n], "n", b, fmt="{:.2f}")
        g.macro(r["pool"]["accuracy"], "IsPoolAccuracy", TAG[n])
        pool, top = r["pool"], str(r["budgets"][-1])
        g.macro(thousands(pool["n"]), "IsPool", TAG[n])
        g.macro(thousands(pool["errors"]), "IsErrors", TAG[n])
        g.macro(thousands(pool["items_with_zero_reported_uncertainty"]), "IsZeroUncertainty", TAG[n])
        g.macro(thousands(pool["errors_with_zero_reported_uncertainty"]), "IsErrorsZeroUncertainty", TAG[n])
        g.macro(pool["share_of_sampling_probability_on_errors"], "IsProposalOnErrors", TAG[n], fmt="{:.2f}")
        g.macro(pool["error_rate"], "IsErrorRate", TAG[n], fmt="{:.2f}")
        half = r["alpha_variance_ratio_uniform_over_importance"].get("0.5", {}).get(top)
        if half is not None:
            g.macro(half, "IsVarRatioHalf", TAG[n], fmt="{:.2f}")
        rows.append(f"{tex_escape(short(r['title']))} & {thousands(r['pool']['n'])} & {r['pool']['accuracy']:.3f} & " + " & ".join(cells) + " \\\\")
    g.src = f"results/{g.mode}/*/importance/importance.json"
    any_ = next(iter(res.values()))
    g.macro(thousands(any_["trials"]), "ISTrials")
    g.macro(any_["epsilon"], "ISEpsilon", fmt="{:.1f}")
    g.macro(any_["alpha"], "ISAlpha", fmt="{:g}")
    g.macro(", ".join(str(b) for b in any_["budgets"]), "ISBudgets")
    g.macro(any_["budgets"][-1], "ISMaxBudget", fmt="{:d}")
    g.macro(min(float(a) for a in any_["alpha_sensitivity"]), "ISAlphaSqrt", fmt="{:g}")
    g.macro(sum(1 for r in res.values() if (r["alpha_variance_ratio_uniform_over_importance"]["0.5"][str(r["budgets"][-1])] or 0) < 1),
            "IsDatasetsSqrtBelowOne", fmt="{:d}")
    g.macro(sum(1 for r in res.values() if r["variance_ratio_uniform_over_importance"][str(r["budgets"][-1])] > 1), "IsDatasetsGuidedWins", fmt="{:d}")
    g.macro(max(abs(r["samplers"][s][str(b)]["bias"]) for r in res.values() for s in ("uniform", "importance") for b in r["budgets"]),
            "IsMaxAbsBias", fmt="{:.3f}")
    head = ("\\begin{tabular}{lrr" + "ccc" * len(shown) + "}\n\\toprule\n & & & " +
            " & ".join(f"\\multicolumn{{3}}{{c}}{{$n = {b}$}}" for b in shown) + " \\\\\n" +
            "".join(f"\\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(shown))) +
            "\nDataset & Pool & Accuracy & " + " & ".join("uniform & guided & ratio" for _ in shown) + " \\\\\n\\midrule\n")
    g.table("tab_importance", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, len(res), figsize=(2.2 * len(res), 2.5))
    for ax, (n, r) in zip(axes, res.items()):
        b = r["budgets"]
        ax.plot(b, [r["samplers"]["uniform"][str(x)]["rmse"] for x in b], "o-", ms=3, c="C0", lw=1, label="uniform")
        ax.plot(b, [r["samplers"]["importance"][str(x)]["rmse"] for x in b], "o-", ms=3, c="C3", lw=1, label="confidence-guided")
        ax.set_xscale("log")
        ax.set_xticks(b, [str(x) for x in b])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_title(short(r["title"]), fontsize=8)
        ax.set_xlabel("label budget $n$", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("RMSE of accuracy estimate", fontsize=8)
    axes[0].legend(fontsize=7)
    g.savefig(fig, "importance")


# ---- explanations ------------------------------------------------------------------------------

def explain_part(g: Generated) -> None:
    res = {n: g.read(n, "explain", "explain") for n in config.DATASETS}
    cfg = config.load("explain")
    rows, total_requests, total_usd = [], 0, 0.0
    for n, r in res.items():
        g.use(config.RESULTS / g.mode / n / "explain" / "explain.json")
        f = r["faithfulness"]
        k3s, k3l = f["kernel_shap"]["3"], f["lime"]["3"]
        rows.append(f"{tex_escape(short(r['title']))} & {r['unit']} & {r['mean_units']:.0f} & {k3s['top_drop']:.3f} & {k3l['top_drop']:.3f} & "
                    f"{k3s['rand_drop']:.3f} & {k3s['top_flip']:.2f} & {k3s['rand_flip']:.2f} & {r['agreement']['kendall_tau_mean']:.2f} & "
                    f"{r['agreement']['top3_overlap_mean']:.2f} & {thousands(r['jev_requests'])} \\\\")
        for k in map(str, cfg["deletion_k"]):
            g.macro(f["kernel_shap"][k]["top_drop"], "ExDropShap", TAG[n], "k", k)
            g.macro(f["lime"][k]["top_drop"], "ExDropLime", TAG[n], "k", k)
            g.macro(f["kernel_shap"][k]["rand_drop"], "ExDropRandom", TAG[n], "k", k)
            g.macro(f["kernel_shap"][k]["top_flip"], "ExFlipShap", TAG[n], "k", k, fmt="{:.2f}")
            g.macro(f["kernel_shap"][k]["rand_flip"], "ExFlipRandom", TAG[n], "k", k, fmt="{:.2f}")
        g.macro(r["agreement"]["kendall_tau_mean"], "ExTau", TAG[n], fmt="{:.2f}")
        g.macro(r["n_examples"], "ExExamples", TAG[n], fmt="{:d}")
        g.macro(thousands(r["selection"]["eligible"]), "ExEligible", TAG[n])
        g.macro(thousands(r["jev_requests"]), "ExRequests", TAG[n])
        g.macro(r["failed_requests"], "ExFailed", TAG[n], fmt="{:d}")
        g.macro(r["agreement"]["top3_overlap_mean"], "ExTopThree", TAG[n], fmt="{:.2f}")
        g.macro(r["mean_units"], "ExMeanUnits", TAG[n], fmt="{:.0f}")
        g.macro(r["lime_r2_mean"], "ExLimeRTwo", TAG[n], fmt="{:.2f}")
        if r.get("positive_given_empty_input") is not None:
            g.macro(r["positive_given_empty_input"], "ExEmptyInputPositive", TAG[n], fmt="{:.2f}")
        total_requests += r["jev_requests"]
        total_usd += r.get("cost_usd", {}).get("usd", 0.0)
    g.src = f"results/{g.mode}/*/explain/explain.json"
    g.macro(thousands(total_requests), "ExRequestsTotal")
    g.macro(thousands(min(r["jev_requests"] for r in res.values())), "ExRequestsMin")
    g.macro(thousands(max(r["jev_requests"] for r in res.values())), "ExRequestsMax")
    g.macro(total_usd, "ExCostUSD", fmt="{:.2f}")
    g.macro(max(r["shap_additivity_error"]["max"] for r in res.values()), "ExAdditivityMax", fmt="{:.0e}")
    g.macro(min(r["lime_r2_mean"] for r in res.values()), "ExLimeRTwoMin", fmt="{:.2f}")
    g.macro(max(r["lime_r2_mean"] for r in res.values()), "ExLimeRTwoMax", fmt="{:.2f}")
    g.use(config.CONFIGS / "explain.yaml")
    g.macro(cfg["shap_nsamples"], "ShapCoalitions", fmt="{:d}")
    g.macro(cfg["lime_samples"], "LimeSamples", fmt="{:d}")
    g.macro(cfg["min_words"], "ExMinWords", fmt="{:d}")
    g.macro(cfg["max_words"], "ExMaxWords", fmt="{:d}")
    g.macro(cfg["random_draws"], "ExRandomDraws", fmt="{:d}")
    from ..experiments.explain import requests_per_example
    g.macro(requests_per_example(cfg), "ExRequestsPerExample", fmt="{:d}")
    head = ("\\begin{tabular}{llrcccccccr}\n\\toprule\n & & & \\multicolumn{3}{c}{Probability drop, $k=3$} & "
            "\\multicolumn{2}{c}{Label flips, $k=3$} & \\multicolumn{2}{c}{Agreement} & \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
            "Dataset & Unit & Units & SHAP & LIME & random & SHAP & random & $\\tau$ & top-3 & Jev requests \\\\\n\\midrule\n")
    g.table("tab_explain", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, len(res), figsize=(2.2 * len(res), 2.5), sharey=True)
    for ax, (n, r) in zip(axes, res.items()):
        ks, f = [int(k) for k in r["faithfulness"]["kernel_shap"]], r["faithfulness"]
        ax.plot(ks, [f["kernel_shap"][str(k)]["top_drop"] for k in ks], "o-", ms=3, c="C0", lw=1, label="KernelSHAP top-$k$")
        ax.plot(ks, [f["lime"][str(k)]["top_drop"] for k in ks], "o-", ms=3, c="C1", lw=1, label="LIME top-$k$")
        ax.plot(ks, [f["kernel_shap"][str(k)]["rand_drop"] for k in ks], "s--", ms=3, c="grey", lw=1, label="random $k$")
        ax.set_title(short(r["title"]), fontsize=8)
        ax.set_xlabel(f"{r['unit']}s deleted ($k$)", fontsize=8)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("drop in P(predicted class)", fontsize=8)
    axes[0].legend(fontsize=7)
    g.savefig(fig, "faithfulness")
    for n in ("german", "sst2"):
        if n in res:
            shutil.copyfile(config.RESULTS / g.mode / n / "explain" / "attributions.pdf", g.out / "figs" / f"attributions_{n}.pdf")


# ---- appendix ----------------------------------------------------------------------------------

LISTING = "\\begin{lstlisting}\n%s\n\\end{lstlisting}\n"


def payload_appendix(g: Generated) -> None:
    parts = []
    for n in config.DATASETS:
        path = config.RESULTS / "live" / "payload_examples" / f"{n}.json"
        if g.mode != "live":
            parts.append(f"\\paragraph{{{tex_escape(datasets.load(n).title)}.}} No logged exchange exists in mock mode.\n")
            continue
        if not path.exists():
            raise RuntimeError(f"{n} has no exported logged exchange ({path}); run `python -m ciayn payload-examples --mode live`")
        rec = json.loads(path.read_text())
        ex = rec["exchange"]
        g.use(path)
        ans = next(iter(ex["response"]["answers"].values()))
        g.macro(rec["example_id"], "PayloadExample", TAG[n])
        g.macro(ans["choice"], "PayloadChoice", TAG[n])
        g.macro(ans["confidence"], "PayloadConfidence", TAG[n], fmt="{:.2f}")
        g.macro(ex["response"]["usage"]["input_tokens"], "PayloadTokens", TAG[n], fmt="{:d}")
        meta = (f"Example \\texttt{{{tex_escape(rec['example_id'])}}}, gold label \\texttt{{{tex_escape(rec['gold_label'])}}}; "
                f"ledger number {ex['ledger_n']}, sent {breakable(tex_escape(ex['ts']))}; "
                f"\\texttt{{{tex_escape(ex['method'])} {breakable(tex_escape(ex['url']))}}}; "
                f"SDK \\texttt{{typesafe-sdk {tex_escape(ex['sdk_version'])}}}; requested model \\texttt{{{tex_escape(ex['requested_model'])}}}, "
                f"reported model \\texttt{{{tex_escape(ex['reported_model'])}}}; HTTP status {ex['status']}; "
                f"latency {ex['latency_s']:.2f}\\,s.")
        parts.append(f"\\subsection*{{{tex_escape(datasets.load(n).title)}}}\n{meta}\n\n\\noindent\\textbf{{Request body as sent}}\n" +
                     LISTING % json.dumps(ex["request"], indent=1, ensure_ascii=True) +
                     "\\noindent\\textbf{Response body as received}\n" +
                     LISTING % json.dumps(ex["response"], indent=1, ensure_ascii=True))
    g.table("appendix_payloads", "\n".join(parts))
    names = ("AuditUnlogged", "AuditUnloggedCached", "AuditRegeneratedMatches")
    if g.mode == "live":
        audit = config.RESULTS / "live" / "payload_audit.json"
        if not audit.exists():
            raise RuntimeError(f"missing {audit}; run `python -m ciayn payload-audit --mode live`")
        a = json.loads(audit.read_text())
        g.use(audit)
        for name, key in zip(names, ("ledger_entries_without_a_logged_exchange", "cached_responses_without_a_logged_exchange",
                                     "regenerated_requests_matching_a_stored_cache_key")):
            g.macro(a[key], name, fmt="{:d}")
    else:
        g.use(config.RESULTS / g.mode / "provenance.json")
        for name in names:
            g.macro(0, name, fmt="{:d}")


def dataset_appendix(g: Generated, dss: dict) -> None:
    from ..backends.jev import instructions
    from ..data.cards import _source

    cfg = config.load("datasets")
    parts = []
    for n, ds in dss.items():
        counts = np.bincount(ds.labels(ds.eval_ids), minlength=len(ds.classes))
        classes = "; ".join(f"\\texttt{{{tex_escape(c)}}} ({tex_escape(ds.criteria[c])}, {counts[i]} evaluation examples)"
                            for i, c in enumerate(ds.classes))
        r0 = ds.repeats[0]
        if ds.protocol == "repeated":
            split = (f"{len(ds.repeats)} stratified repeats with seeds 0--{len(ds.repeats) - 1}; train {len(r0['train'])}, "
                     f"validation {len(r0['val'])}, test {len(r0['test'])} in each; {len(ds.eval_ids)} distinct rows are a test row in some repeat")
            prep = (f"one \\texttt{{name: value}} line per feature ({len(ds.features)} features), categorical codes decoded; a missing "
                    f"value is written as \\texttt{{{cfg['missing_value']}}} ({int(ds.frame.isna().sum().sum())} missing values in this dataset)")
        else:
            split = (f"official {cfg[n]['eval_split']} split ({thousands(len(r0['test']))} examples, all used); validation "
                     f"{thousands(len(r0['val']))} drawn stratified from the official training split with seed {cfg['seed']}; training {thousands(len(r0['train']))}")
            prep = f"the text itself, with {tex_escape(cfg['text_normalisation'])}; no example is empty"
        parts.append(f"\\paragraph{{{tex_escape(ds.title)}.}} \\emph{{Source:}} {breakable(tex_escape(_source(ds).replace('`', '')))}. "
                     f"\\emph{{Classes, in the order sent:}} {classes}. "
                     f"\\emph{{Positive class:}} {tex_escape(ds.positive) if ds.positive else 'not applicable'}. "
                     f"\\emph{{Splits:}} {split}. \\emph{{Input:}} {prep}. "
                     f"\\emph{{Instructions:}} ``{tex_escape(instructions(ds))}''\n")
    g.table("appendix_datasets", "\n".join(parts))


def per_class_appendix(g: Generated) -> None:
    parts = []
    for n in config.DATASETS:
        base = config.RESULTS / g.mode / n / "metrics"
        tabs = {m: pd.read_csv(base / f"per_class_{m}.csv") for m in config.MODELS if (base / f"per_class_{m}.csv").exists()}
        j = tabs["jev"]
        rows = []
        for i, (c, sup) in enumerate(zip(j["class"], j["support"])):
            cells = " & ".join(f"{t['precision'][i]:.2f} & {t['recall'][i]:.2f} & {t['f1'][i]:.2f}" for t in tabs.values())
            rows.append(f"{tex_escape(c)} & {int(sup)} & {cells} \\\\")
        title = tex_escape(datasets.load(n).title)
        short_name = {"jev": "Jev", "qwen": "Qwen3.6-27B", "xgb": "XGBoost", "bert": "BERT"}
        groups = " & ".join(f"\\multicolumn{{3}}{{c}}{{{short_name[m]}}}" for m in tabs)
        rules = "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(tabs)))
        head = (f"\\toprule\n & & {groups} \\\\\n{rules}\nClass & Support & " + " & ".join("P & R & F1" for _ in tabs) +
                " \\\\\n\\midrule\n")
        spec = "lr" + "ccc" * len(tabs)
        note = " Support and metrics are pooled over the split repeats." if datasets.load(n).protocol == "repeated" else ""
        for m in config.MODELS:
            if m not in tabs:
                note += f" Not available: {ROW_LABEL[m]}."
        if len(rows) > 10:
            parts.append("{\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{longtable}{" + spec + "}\n\\caption{Per-class precision (P), "
                         "recall (R) and F1 on " + title + "." + note + "}\\\\\n" + head + "\\endfirsthead\n" + head +
                         "\\endhead\n\\bottomrule\n\\endfoot\n" + "\n".join(rows) + "\n\\end{longtable}\n}\n")
        else:
            parts.append("\\begin{table}[H]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\fittable{\\begin{tabular}{" + spec +
                         "}\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\caption{Per-class precision (P), recall (R) "
                         "and F1 on " + title + "." + note + "}\n\\end{table}\n")
    g.table("appendix_perclass", "\n".join(parts))


# ---- the added models: configuration macros -------------------------------------------------------

def models_part(g: Generated) -> None:
    enc, llm, esc = config.load("encoder"), config.load("llm"), config.load("escalation")
    g.use(config.CONFIGS / "encoder.yaml")
    g.macro(enc["model_id"], "BertModelId")
    g.macro(enc["revision"][:12], "BertRevision")
    g.macro(enc["image"], "BertImage")
    g.macro(" and ".join(f"{lr:g}" for lr in enc["grid"]["learning_rate"]), "BertLearningRates")
    g.macro(enc["max_epochs"], "BertMaxEpochs", fmt="{:d}")
    g.macro(enc["batch_size"], "BertBatchSize", fmt="{:d}")
    for n, v in enc["max_length"].items():
        g.macro(v, "BertMaxLength", TAG[n], fmt="{:d}")
    g.use(config.CONFIGS / "llm.yaml")
    g.macro(llm["model_id"], "QwenModelId")
    g.macro(llm["revision"][:12], "QwenRevision")
    g.macro(llm["infer"]["image"], "QwenImage")
    g.macro(llm["infer"]["image"].split(":")[1].split("-")[0], "QwenVllmVersion")      # the container's vLLM version
    g.macro(llm["thinking"]["subsample_per_dataset"], "ThinkSubsampleN", fmt="{:d}")
    g.macro(thousands(llm["thinking"]["max_thinking_tokens"]), "ThinkMaxTokens")
    g.macro(llm["latency"]["sample_per_dataset"], "QwenLatencyN", fmt="{:d}")
    g.macro(llm["cross_check_items"], "QwenCrossCheckN", fmt="{:d}")
    g.use(config.CONFIGS / "escalation.yaml")
    g.macro(esc["validation_items"]["agnews"], "EscValItemsNew", fmt="{:d}")
    for n in config.DATASETS:                     # what the fine-tuning and the Qwen run recorded
        run = config.RESULTS / g.mode / n / "bert" / "run.json"
        if run.exists():
            r = json.loads(run.read_text())
            if r.get("selected"):
                g.use(run)
                g.macro(r["selected"]["epochs"], "BertEpochs", TAG[n], fmt="{:d}")
                if r.get("gpu"):
                    g.macro(r["gpu"], "BertGpu")
                g.macro(sum(r["repeats"]["0"]["truncated"].values()), "BertTruncated", TAG[n], fmt="{:d}")
        run = config.RESULTS / g.mode / n / "qwen" / "run.json"
        if run.exists():
            r = json.loads(run.read_text())
            cc = (r.get("timing") or {}).get("cross_check")
            if cc:
                g.use(run)
                g.macro(cc["max_abs_probability_difference"], "QwenCrossCheckDiff", TAG[n], fmt="{:.4f}")
            ccheck = (r.get("timing") or {}).get("cache_check")
            if ccheck:
                g.use(run)
                g.macro(ccheck["max_abs_logprob_difference"], "QwenCacheCheck", TAG[n], fmt="{:.3f}")
                g.macro((r.get("timing") or {}).get("shared_prefix_tokens", 0), "QwenSharedPrefix", TAG[n], fmt="{:d}")
            if r.get("versions"):
                g.use(run)
                g.macro(r["versions"].get("transformers", "not recorded"), "QwenTransformersVersion")
                g.macro(r["versions"].get("torch", "not recorded"), "QwenTorchVersion")
                g.macro(r.get("instance_type", "not recorded"), "QwenScoringInstance")
                g.macro(", ".join(sorted(set(r.get("gpus") or []))) or "not recorded", "QwenGpu")
    if "BertGpu" not in g.macros:
        g.macros["BertGpu"], g.sources["BertGpu"] = "not recorded", "no BERT run in these results"
    for name in ("QwenTransformersVersion", "QwenTorchVersion", "QwenScoringInstance", "QwenGpu"):
        if name not in g.macros:                                                     # no live run recorded it
            g.macros[name], g.sources[name] = "not recorded", "no Qwen run in these results"
    path = config.RESULTS / g.mode / "xgb_rerun_check.json"
    if path.exists():
        g.use(path)
        g.macro("identical" if json.loads(path.read_text())["all_identical"] else "not identical", "XgbRerunPredictions")
    else:
        g.macros["XgbRerunPredictions"], g.sources["XgbRerunPredictions"] = "not compared", "no rerun in these results"
    path = config.RESULTS / g.mode / "qwen_thinking_projection.json"       # thinking on every item: projected, and whether it was run
    names = ("QwenThinkingAllProjectedUSD", "QwenThinkingAllProjectedHours", "QwenThinkingAllFits")
    if path.exists():
        pr = json.loads(path.read_text())
        g.use(path)
        g.macro(pr["projected_usd"], names[0], fmt="{:.0f}")
        g.macro(pr["projected_seconds"] / 3600, names[1], fmt="{:.1f}")
        g.macro("fits" if pr["fits"] else "does not fit", names[2])
    else:
        for name in names:
            g.macros[name], g.sources[name] = "not projected", "no projection in these results"


# ---- comparison by training labels, cost and latency --------------------------------------------

def _usd(v) -> str:
    if v is None:
        return "--"
    if 0 < v < 0.0001:
        return "$<$0.0001"
    return f"{v:.4f}" if v < 0.01 else f"{v:.3f}" if v < 1 else f"{v:.2f}"


def _secs(v) -> str:
    if v is None:
        return "--"
    if v < 1e-5:
        return f"{v * 1e6:.1f}\\,$\\mu$s"
    return f"{v * 1000:.2f}\\,ms" if v < 0.1 else f"{v:.2f}\\,s"


def _val(h: dict) -> str:
    """A value with its standard deviation over repeats, or alone for a single split."""
    return unc(h) if h.get("n_repeats", 1) > 1 else f"{h['value']:.3f}"


def comparison_part(g: Generated, metrics: dict) -> dict:
    res = {n: g.read(n, "comparison", "comparison") for n in config.DATASETS}
    rows = []
    for n, r in res.items():
        g.use(config.RESULTS / g.mode / n / "comparison" / "comparison.json")
        present = [m for m in config.MODELS if m in r["costs"]]
        for i, m in enumerate(present):
            c, h = r["costs"][m], metrics[n]["models"][m]["headline"]
            lb = metrics[n]["label_budget"][m]
            first = f"\\multirow{{{len(present)}}}{{*}}{{{tex_escape(short(r['title']))}}}" if i == 0 else ""
            if m == "jev":
                lat = _secs(c["latency"]["mean_s"])
            elif m == "qwen":
                lt = c["latency"]
                lat = (f"{_secs(lt['single_request_mean_s'])} ({lt['throughput_items_per_s']:.1f}/s)"
                       if lt.get("throughput_items_per_s") else "--")
            else:
                lat = _secs(c["latency"]["per_item_s"])
            per1k = _usd(c.get("usd_per_1k"))
            if m == "qwen" and c.get("usd_per_1k_inference_only") is not None:
                per1k += f" ({_usd(c['usd_per_1k_inference_only'])})"
            train = "--" if m in config.ZERO_SHOT else _usd(c.get("training_usd"))
            rows.append(f"{first} & {SHORT[m]} & {thousands(lb['training'])} & {h['macro_f1']['value']:.3f} & "
                        f"{train} & {per1k} & {lat} \\\\")
            if m == "qwen" and c.get("usd_per_1k") and r["costs"]["jev"].get("usd_per_1k"):
                g.macro(c["usd_per_1k"] / r["costs"]["jev"]["usd_per_1k"], "CostRatioQwenJev", TAG[n], fmt="{:.0f}")
            if m in config.SUPERVISED and m in r["f1_by_labels"]:
                jev_f1 = metrics[n]["models"]["jev"]["headline"]["macro_f1"]["value"]
                reach = next((p["n_train"] for p in r["f1_by_labels"][m] if p["macro_f1"] >= jev_f1), None)
                g.macro(thousands(reach) if reach is not None else "no size tried", "LabelsToMatchJev", MODEL_TAG[m], TAG[n])
            if c.get("usd_per_1k") is not None:
                g.macro(c["usd_per_1k"], "CostPerK", MODEL_TAG[m], TAG[n], fmt="{:.3f}")
            if m == "qwen" and c.get("usd_per_1k_inference_only") is not None:
                g.macro(c["usd_per_1k_inference_only"], "CostPerKInferenceOnly", "Qwen", TAG[n], fmt="{:.3f}")
                g.macro(c["latency"]["throughput_items_per_s"], "QwenThroughput", TAG[n], fmt="{:.1f}")
                g.macro(c["latency"]["single_request_mean_s"], "QwenLatency", TAG[n], fmt="{:.2f}")
            if m in config.SUPERVISED and c.get("training_usd") is not None:
                g.macro(c["training_usd"], "TrainUSD", MODEL_TAG[m], TAG[n], fmt="{:.2f}")
            if m in config.SUPERVISED and c["latency"].get("per_item_s") is not None:
                g.macro(c["latency"]["per_item_s"] * 1000, "InferMs", MODEL_TAG[m], TAG[n], fmt="{:.2f}")
            if m == "jev" and c["latency"]["mean_s"] is not None:
                g.macro(c["latency"]["mean_s"], "JevLatency", TAG[n], fmt="{:.2f}")
        if n != config.DATASETS[-1]:
            rows.append("\\addlinespace")
    g.table("tab_costs", booktabs("llrcrll", "Dataset & Model & Labels & Macro-F1 & Training & Per 1,000 & Latency", rows))

    fig, axes = plt.subplots(1, len(res), figsize=(2.2 * len(res), 2.6))
    short_label = {"jev": "Jev", "qwen": "Qwen3.6-27B", "xgb": "XGBoost", "bert": "BERT"}
    for ax, (n, r) in zip(axes, res.items()):
        for m, pts in r["f1_by_labels"].items():
            if m in config.SUPERVISED:
                ax.errorbar([p["n_train"] for p in pts], [p["macro_f1"] for p in pts], yerr=[p["sd"] for p in pts], marker="o", ms=3,
                            c=MODEL_COLOUR[m], capsize=2, lw=1, label=short_label[m])
            else:
                ax.axhline(pts[0]["macro_f1"], c=MODEL_COLOUR[m], ls="--", lw=1, label=f"{short_label[m]}, zero-shot")
        ax.set_xscale("log")
        c = next((pts for m, pts in r["f1_by_labels"].items() if m in config.SUPERVISED), None)
        if c:
            ax.set_xticks([c[0]["n_train"], c[len(c) // 2]["n_train"], c[-1]["n_train"]],
                          [f"{p['n_train']:,}" for p in (c[0], c[len(c) // 2], c[-1])])
            ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_title(short(r["title"]), fontsize=8)
        ax.set_xlabel("training labels", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("macro-F1", fontsize=8)
    axes[0].legend(fontsize=6)
    g.savefig(fig, "f1_by_labels")
    return res


# ---- escalation ----------------------------------------------------------------------------------

def escalation_part(g: Generated) -> dict:
    res = {}
    for n in config.DATASETS:
        path = config.RESULTS / g.mode / n / "escalation" / "escalation.json"
        if path.exists():
            res[n] = g.read(n, "escalation", "escalation")
    rows = []
    for n, r in res.items():
        g.use(config.RESULTS / g.mode / n / "escalation" / "escalation.json")
        s, u = r["summary"], r["unreachable"]
        jv, qv = s["jev_alone"]["macro_f1"], s["qwen_alone"]["macro_f1"]
        g.macro(jv["value"], "EscJevFOne", TAG[n])
        g.macro(qv["value"], "EscQwenFOne", TAG[n])
        g.macro(thousands(u["errors_at_max"]), "EscUnreachable", TAG[n])
        g.macro(thousands(u["errors"]), "EscErrors", TAG[n])
        g.macro(u["share_at_max"], "EscShareAtMax", TAG[n], fmt="{:.2f}")
        g.macro(thousands(r["repeats"][0]["n_validation_labels"]), "EscValLabels", TAG[n])
        if r["threshold_selected"]:
            g.macro(s["macro_f1"]["value"], "EscFOne", TAG[n])
            g.macro(s["accuracy"]["value"], "EscAcc", TAG[n])
            g.macro(s["share_escalated"]["value"], "EscShare", TAG[n], fmt="{:.2f}")
            g.macro(s["threshold"]["value"], "EscThreshold", TAG[n], fmt="{:.2f}")
            cost = r["costs_at_selected"]
            if cost.get("usd_per_1k") is not None:
                g.macro(cost["usd_per_1k"], "EscCostPerK", TAG[n], fmt="{:.3f}")
            if cost.get("mean_latency_s") is not None:
                g.macro(cost["mean_latency_s"], "EscLatency", TAG[n], fmt="{:.2f}")
            sel = (f"{s['threshold']['value']:.2f} & {s['share_escalated']['value']:.2f} & {_val(s['macro_f1'])} & "
                   f"{_usd(cost.get('usd_per_1k'))}")
        else:
            sel = "\\multicolumn{4}{c}{\\emph{no validation items: no threshold selected}}"
        rows.append(f"{tex_escape(short(r['title']))} & {_val(jv)} & {_val(qv)} & {sel} & {thousands(u['errors_at_max'])} of {thousands(u['errors'])} \\\\")
    if res:
        g.table("tab_escalation", booktabs("lccccccr", "Dataset & Jev & Qwen & $t$ & Share & Ensemble & USD/1,000 & Unreachable", rows))
        fig, axes = plt.subplots(1, len(res), figsize=(2.2 * len(res), 2.5), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, (n, r) in zip(axes, res.items()):
            rep = r["repeats"][0]
            c = rep["curve"]
            ax.plot([x["share_escalated"] for x in c], [x["macro_f1"] for x in c], c="k", lw=1)
            ax.scatter([0], [c[0]["macro_f1"]], c=MODEL_COLOUR["jev"], s=14, zorder=3, label="Jev alone")
            ax.scatter([1], [c[-1]["macro_f1"]], c=MODEL_COLOUR["qwen"], s=14, zorder=3, label="Qwen alone")
            if rep.get("at"):
                ax.scatter([rep["at"]["share_escalated"]], [rep["at"]["macro_f1"]], marker="*", s=60, c="C3", zorder=4, label="selected")
            ax.set_title(short(r["title"]), fontsize=8)
            ax.set_xlabel("share escalated", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("macro-F1", fontsize=8)
        axes[0].legend(fontsize=6)
        g.savefig(fig, "escalation")
    return res


# ---- thinking subsample ----------------------------------------------------------------------------

def thinking_part(g: Generated, metrics: dict) -> None:
    rows = []
    for n, m in metrics.items():
        sub = m.get("thinking_subsample")
        if not sub:
            continue
        g.use(config.RESULTS / g.mode / n / "metrics" / "metrics.json")
        think = results_frame(g, n, "qwen_think")
        cells = []
        for mo in ("jev", "qwen", "qwen_think", "xgb", "bert"):
            v = sub["models"].get(mo, {}).get("macro_f1")
            cells.append("--" if v is None else f"{v:.3f}")
            if v is not None:
                g.macro(v, "ThinkSubFOne", MODEL_TAG[mo], TAG[n])
        g.macro(sub["n"], "ThinkSubN", TAG[n], fmt="{:d}")
        tok = think.get("thinking_tokens")
        extra = "-- & --"
        if tok is not None:
            g.use(config.RESULTS / g.mode / n / "qwen_think" / "predictions.parquet")
            g.macro(thousands(round(float(np.mean(tok)))), "ThinkTokens", TAG[n])
            g.macro(float(np.mean(think["thinking_truncated"])), "ThinkTruncated", TAG[n], fmt="{:.2f}")
            g.macro(float(np.mean(think["seconds"])), "ThinkSeconds", TAG[n], fmt="{:.2f}")
            extra = f"{thousands(round(float(np.mean(tok))))} & {np.mean(think['thinking_truncated']):.2f}"
        rows.append(f"{tex_escape(short(m['title']))} & {sub['n']} & " + " & ".join(cells) + f" & {extra} \\\\")
    if rows:
        g.table("tab_thinking", booktabs("lrcccccrr", "Dataset & $n$ & Jev & Qwen, off & Qwen, on & XGBoost & BERT & "
                                         "Thinking tokens & Truncated", rows))


def results_frame(g: Generated, n: str, model: str) -> dict:
    path = config.RESULTS / g.mode / n / model / "predictions.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {c: df[c].to_numpy() for c in df.columns}


# ---- Qwen prompt appendix --------------------------------------------------------------------------

def qwen_appendix(g: Generated) -> None:
    parts = []
    for n in config.DATASETS:
        run = config.RESULTS / g.mode / n / "qwen" / "run.json"
        title = tex_escape(datasets.load(n).title)
        rec = json.loads(run.read_text()).get("rendered_example") if run.exists() else None
        if not rec:
            parts.append(f"\\paragraph{{{title}.}} No rendered Qwen prompt exists for this build"
                         f"{' (mock mode)' if g.mode == 'mock' else ''}.\n")
            continue
        g.use(run)
        g.macro(rec["id"], "QwenExample", TAG[n])
        df = pd.read_parquet(config.RESULTS / g.mode / n / "qwen" / "predictions.parquet").set_index("id")
        ds = datasets.load(n)
        probs = df.loc[rec["id"], results_cols(ds)].to_numpy(float)
        top = np.argsort(-probs)[:5]
        dist = ", ".join(f"\\texttt{{{tex_escape(ds.classes[i])}}} {probs[i]:.3f}" for i in top)
        more = f" (the five largest of {len(ds.classes)})" if len(ds.classes) > 5 else ""
        parts.append(f"\\subsection*{{{title}}}\nExample \\texttt{{{tex_escape(rec['id'])}}}. Prompt with thinking disabled, as "
                     f"rendered by the model's tokenizer inside the job:\n" + LISTING % rec["thinking_disabled"] +
                     f"Qwen3.6-27B's distribution{more}: {dist}.\n")
    g.table("appendix_qwen", "\n".join(parts))


def results_cols(ds) -> list[str]:
    return [f"p{i}" for i in range(len(ds.classes))]


# ---- driver ------------------------------------------------------------------------------------

def is_mock_draft(g: Generated) -> bool:
    """The paper is a watermarked draft if it was asked for in mock mode or consumed any mock-labelled artifact."""
    return g.mode == "mock" or "mock" in g.artifact_modes


def generate(mode: str, out: Path | None = None) -> Generated:
    g = Generated(mode, out)
    shutil.rmtree(g.out, ignore_errors=True)
    (g.out / "figs").mkdir(parents=True)
    provenance_part(g)
    dss = datasets_part(g)
    metrics = metrics_part(g)
    prior_part(g, metrics)
    thresholding_part(g)
    importance_part(g)
    explain_part(g)
    models_part(g)
    comparison_part(g, metrics)
    escalation_part(g)
    thinking_part(g, metrics)
    payload_appendix(g)
    qwen_appendix(g)
    dataset_appendix(g, dss)
    per_class_appendix(g)
    (g.out / "macros.tex").write_text("% generated by `python -m ciayn paper`: do not edit\n" +
                                      "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(g.macros.items())))
    (g.out / "macro_sources.json").write_text(json.dumps({k: {"value": g.macros[k], "source": g.sources[k]}
                                                          for k in sorted(g.macros)}, indent=1))
    mock = is_mock_draft(g)
    (g.out / "mockdraft.tex").write_text("\\def\\mockdraft{}\n" if mock else "% live results only\n")
    (g.out / "mode.json").write_text(json.dumps({"mode": mode, "artifact_modes": sorted(g.artifact_modes), "mockdraft": mock}, indent=1))
    return g


def run(args) -> int:
    g = generate(args.mode)
    print(f"paper ({args.mode}): wrote {len(g.macros)} macros, tables, figures and appendices to {g.out}; "
          f"artifact modes {sorted(g.artifact_modes)}")
    if args.no_build:
        return 0
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    r = subprocess.run(cmd, cwd=config.PAPER, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError("latexmk failed; see paper/main.log")
    print(f"paper ({args.mode}): built {config.PAPER / 'main.pdf'}")
    return 0
