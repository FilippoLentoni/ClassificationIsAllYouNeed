"""Qwen: prompts, option scoring, the job script on a tiny model on CPU, pipeline and run logic. No AWS, no download."""
import json
from pathlib import Path

import numpy as np
import pytest

from ciayn import config, templates
from ciayn.data.serialize import serialize
from ciayn.llm import prompts, thinking_subsample
from ciayn.llm.scripts import qwen_infer


def test_prompt_carries_jevs_state_instructions_and_options_in_order(tmp_path, german, sst2):
    saved = {d.name: {i: {"reasoning_ids": [1], "thinking_truncated": False} for i in thinking_subsample(d)} for d in (german, sst2)}
    out = prompts.write_job([german, sst2], "full-hf", tmp_path / "job", reasoning=saved)
    job = json.loads((out / "job.json").read_text())
    for ds in (german, sst2):
        tpl = templates.require("live", ds)
        rows = [json.loads(x) for x in (out / f"{ds.name}.jsonl").read_text().splitlines()]
        assert {r["id"] for r in rows} == set(ds.jev_ids), "every item Jev predicted, nothing else"
        r = next(x for x in rows if "reasoning_ids" not in x)
        system, user = r["messages"][0]["content"], r["messages"][1]["content"]
        assert user == serialize(ds, r["id"]) and system.startswith(tpl["instructions"])
        pos = [system.index(f"- {name}: {desc}") for name, desc in tpl["criteria"]]
        assert pos == sorted(pos), "options in Jev's order"
        assert all(set(x) <= {"id", "messages", "reasoning_ids", "thinking_truncated", "seconds"} for x in rows), "no labels"
        assert job["options"][ds.name] == [n for n, _ in tpl["criteria"]]
        assert job["sets"][ds.name]["think"] == thinking_subsample(ds) and len(job["sets"][ds.name]["latency"]) == 20
    assert job["revision"] == config.load("llm")["revision"]


def test_smoke_sets(tmp_path, wdbc):
    smoke = prompts.sets(wdbc, "smoke-hf")
    assert len(smoke["nothink"]) == 64 and smoke["nothink"][:50] == thinking_subsample(wdbc) and smoke["think"] == []
    assert set(smoke["nothink"]) <= set(wdbc.ids("test", 0))


def test_distribution_normalises_and_flags_failures():
    p, ok = qwen_infer.distribution([np.log(0.2), np.log(0.6), np.log(0.2)])
    assert ok and np.allclose(p, [0.2, 0.6, 0.2])
    p, ok = qwen_infer.distribution([-1000.0, -1001.0])            # far below zero: still a proper distribution
    assert ok and np.isclose(sum(p), 1) and p[0] > p[1]
    p, ok = qwen_infer.distribution([0.0, float("nan")])
    assert not ok and p == [0.5, 0.5]


class FakeTok:
    """Characters are tokens; `<|im_end|>` is the single token 1."""
    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                out.append(1)
                i += len("<|im_end|>")
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_an_option_that_is_a_prefix_of_another_is_a_complete_answer():
    opts = qwen_infer.option_ids(FakeTok(), ["card", "card_arrival"])
    assert opts[0][-1] == 1 and opts[1][-1] == 1 and opts[0][:-1] == opts[1][:4]

    def logprob(prefix, tok):          # a model that wants to continue "card" with "_arrival"
        if tok == 1:
            return -5.0 if "".join(map(chr, prefix)) == "card" else -0.01
        return -0.01 if chr(tok) == "_" else 0.0

    sums = [sum(logprob(o[:t], o[t]) for t in range(len(o))) for o in opts]
    p, _ = qwen_infer.distribution(sums)
    assert p[1] > 0.99, "without the end token, 'card' would share the mass of every longer option"


# ---- the job script with the transformers engine on a tiny random model --------------------------

def tiny_qwen(path: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config
    words = ["good", "bad", "card", "card_arrival", "movie", "Options:", "-", "yes", "no", "Answer", "system", "user",
             "assistant", "the"]
    special = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "[UNK]", "[PAD]"]
    vocab = {w: i for i, w in enumerate(special + words)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|> assistant {% if not enable_thinking %}<think> </think> "
                         "{% endif %}{% endif %}")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      intermediate_size=64, max_position_embeddings=256)
    import torch
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    return model, tok


def test_cached_option_scoring_equals_full_recomputation():
    import torch
    model, tok = tiny_qwen(None)
    eng = qwen_infer.HfEngine("unused", {"dtype": "float32", "option_chunk": 2}, model=model, tok=tok)
    prefix = tok.encode("<|im_start|> user good movie <|im_end|> <|im_start|> assistant", add_special_tokens=False)
    opts = qwen_infer.option_ids(tok, ["good", "bad", "card", "card_arrival"])
    got = eng.score([prefix], opts)[0]
    with torch.no_grad():
        for j, o in enumerate(opts):
            ids = torch.tensor([prefix + o])
            lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
            want = sum(float(lp[len(prefix) + t - 1, o[t]]) for t in range(len(o)))
            assert got[j] == pytest.approx(want, abs=1e-4), j


def test_job_script_writes_distributions_latency_and_thinking(tmp_path):
    model, tok = tiny_qwen(None)
    src = tmp_path / "prompts"
    src.mkdir()
    items = [{"id": f"x{i}", "messages": [{"role": "system", "content": "Options: - good - bad Answer"},
                                         {"role": "user", "content": w}]} for i, w in enumerate(["good movie", "bad movie", "the"])]
    (src / "toy.jsonl").write_text("\n".join(json.dumps(x) for x in items))
    job = {"datasets": ["toy"], "options": {"toy": ["good", "bad"]}, "example_id": {"toy": "x0"}, "dtype": "float32",
           "sets": {"toy": {"nothink": ["x0", "x1", "x2"], "think": ["x0", "x1"], "latency": ["x2"]}},
           "max_thinking_tokens": 3, "items_per_call": 2}
    (src / "job.json").write_text(json.dumps(job))
    eng = qwen_infer.HfEngine("unused", job, model=model, tok=tok)
    run = qwen_infer.main(src, "unused", tmp_path / "out", engine=eng)
    rows = [json.loads(x) for x in (tmp_path / "out" / "toy.nothink.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["x0", "x1", "x2"] and all(np.isclose(sum(r["p"]), 1) and r["valid"] for r in rows)
    think = [json.loads(x) for x in (tmp_path / "out" / "toy.think.jsonl").read_text().splitlines()]
    assert len(think) == 2 and all(1 <= r["thinking_tokens"] <= 4 for r in think)
    info = run["datasets"]["toy"]
    assert len(info["latency_s"]) == 1 and info["nothink"]["items"] == 3 and info["think"]["items"] == 2
    rendered = json.loads((tmp_path / "out" / "rendered_examples.json").read_text())
    assert rendered["toy"]["id"] == "x0" and "<think> </think>" in rendered["toy"]["thinking_disabled"]
    assert run["engine"] == "transformers" and run["load_s"] >= 0


# ---- pipeline and CLI (FakeAws) ----------------------------------------------------------------

def test_pipeline_definition_is_deterministic_and_records_the_substitution():
    from ciayn.llm import pipeline
    a = pipeline.definition("us-east-2", "bucket", "arn:role", 5400)
    assert a == pipeline.definition("us-east-2", "bucket", "arn:role", 5400)
    stage, cond = a["Steps"]
    assert stage["Type"] == "Processing" and cond["Type"] == "Condition"
    infer = cond["Arguments"]["IfSteps"][0]
    assert infer["Type"] == "Training" and infer["Arguments"]["StoppingCondition"]["MaxRuntimeInSeconds"] == 5400
    assert infer["Arguments"]["AlgorithmSpecification"]["TrainingImage"].endswith(config.load("llm")["infer"]["image"])
    assert config.load("llm")["revision"] in pipeline.weights_uri("bucket")
    assert "training job" in pipeline.SUBSTITUTION and "quota" in pipeline.SUBSTITUTION


class QwenAws:
    """FakeAws plus S3 reads, with a canned job result in the checkpoint location."""
    files: dict = {}
    staged = False

    def __init__(self, region):
        from fakes import FakeAws
        self.inner = FakeAws(region)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def exists(self, bucket, key):
        return QwenAws.staged

    def list_keys(self, bucket, prefix):
        return [f"{prefix}{k}" for k in QwenAws.files]

    def download(self, bucket, key, path):
        Path(path).write_bytes(QwenAws.files[key.rsplit("/", 1)[-1]])


def canned_run(dss, kind="smoke", per_item=0.5):
    run = {"engine": "vllm", "load_s": 300.0, "datasets": {}}
    files = {}
    for ds in dss:
        s = prompts.sets(ds, kind)
        run["datasets"][ds.name] = {"nothink": {"items": len(s["nothink"]), "seconds": per_item * len(s["nothink"])},
                                    "latency_s": [0.2] * len(s["latency"])}
        if s["think"]:
            run["datasets"][ds.name]["think"] = {"items": len(s["think"]), "seconds": 4 * per_item * len(s["think"])}
        k = len(ds.classes)
        for variant in ("nothink", "think"):
            files[f"{ds.name}.{variant}.jsonl"] = "\n".join(json.dumps({"id": i, "valid": True, "p": [1 / k] * k, "seconds": 0.1,
                                                                        "prompt_tokens": 50}) for i in s[variant]).encode()
    files["run.json"] = json.dumps(run).encode()
    return files


def test_stage_does_nothing_when_the_weights_are_there(approved, capsys):
    from conftest import stage_args
    from ciayn.llm import cli
    QwenAws.staged = True
    assert cli.stage(stage_args("live"), make=QwenAws) == 0
    assert "nothing to do" in capsys.readouterr().out
    from fakes import FakeAws
    assert FakeAws.log == []


def _templates():
    import shutil
    shutil.copytree(Path(__file__).resolve().parents[1] / "results" / "live" / "templates", config.results_dir("live") / "templates")


def test_two_jobs_vllm_reasons_and_transformers_scores():
    from ciayn.data import datasets
    agnews, sst2 = datasets.load("agnews"), datasets.load("sst2")
    assert all(prompts.engine_of(n) == "transformers" for n in config.DATASETS)
    for ds in (agnews, sst2):
        full, hf = prompts.sets(ds, "full"), prompts.sets(ds, "full-hf")
        assert full["nothink"] == [] and full["latency"] == [] and full["think"] == thinking_subsample(ds), "vLLM only reasons"
        assert hf["nothink"] == prompts.all_items(ds) and hf["think"] == thinking_subsample(ds) and len(hf["latency"]) == 20
        assert prompts.think_mode(ds, "full") == "generate" and prompts.think_mode(ds, "full-hf") == "given"
    assert len(prompts.sets(agnews, "full-hf")["nothink"]) == 7600 + 500 and len(thinking_subsample(agnews)) == 50


def test_full_hf_job_needs_the_saved_reasoning(tmp_path, approved):
    from ciayn.data import datasets
    _templates()
    ds = datasets.load("wdbc")
    with pytest.raises(RuntimeError, match="reasoning generated by the vLLM job is missing"):
        prompts.write_job([ds], "full-hf", tmp_path / "a")
    saved = {"wdbc": {i: {"reasoning_ids": [7, 8], "thinking_truncated": False, "seconds": 1.0} for i in thinking_subsample(ds)}}
    out = prompts.write_job([ds], "full-hf", tmp_path / "b", reasoning=saved)
    job = json.loads((out / "job.json").read_text())
    rows = {r["id"]: r for r in map(json.loads, (out / "wdbc.jsonl").read_text().splitlines())}
    assert job["engine"] == "transformers" and job["think_mode"] == {"wdbc": "given"} and "items_per_call" not in job
    assert job["datasets"] == ["wdbc"]
    assert rows[thinking_subsample(ds)[0]]["reasoning_ids"] == [7, 8]


def test_smokes_project_both_jobs_and_run_is_refused_until_they_fit(approved, monkeypatch):
    from conftest import stage_args
    from fakes import FakeAws
    from ciayn.baseline import cli as aws_cli
    from ciayn.data import datasets
    from ciayn.llm import cli
    _templates()
    dss = [datasets.load(n) for n in config.DATASETS]
    # a vLLM smoke run that measured scoring and reasoning (as the first live smoke run did)
    run = {"engine": "vllm", "load_s": 300.0,
           "datasets": {d.name: {"nothink": {"items": 20, "seconds": 10.0}, "think": {"items": 5, "seconds": 55.0}, "latency_s": [0.5]}
                        for d in dss}}
    QwenAws.files = {"run.json": json.dumps(run).encode()}
    aws_cli.record_attempt({"model": "qwen", "kind": "smoke", "status": "Succeeded", "region": "us-east-2", "dataset": "all",
                            "instance_type": "ml.g5.12xlarge", "execution_arn": "arn:smoke", "billable_seconds": 900.0,
                            "cost_usd": 1.8, "checkpoints": "s3://b/checkpoints/qwen/s/", "steps": {"Infer": {"billable_seconds": 900.0}}})
    assert cli.smoke(stage_args("live", engine="vllm"), make=QwenAws) == 0 and FakeAws.log == [], "the existing smoke run is reused"
    proj = json.loads((config.results_dir("live") / "qwen_projection.json").read_text())
    assert not proj["complete"] and not proj["fits"] and proj["vllm"]["kind"] == "full"
    per = proj["vllm"]["datasets"]["sst2"]
    assert per["think_s_per_item"] == pytest.approx(11.0 - 0.5), "reasoning only: the scoring part runs in the other job"
    assert per["seconds"] == pytest.approx(10.5 * 50)
    with pytest.raises(RuntimeError, match="no complete cost projection"):
        cli.run_live(stage_args("live", thinking="subsample"), make=QwenAws)
    QwenAws.files = canned_run(dss, "smoke-hf", per_item=2.0)
    LIMIT = "ResourceLimitExceeded: account-level service limit 'ml.g6e.12xlarge for training job usage' is 0 Instances"
    FakeAws.outcomes[("us-east-2", "ml.g6e.12xlarge")] = LIMIT
    (config.results_dir("live") / "llm_combination.json").write_text(json.dumps({"region": "us-east-2", "instance_type": "ml.g6e.12xlarge"}))
    assert cli.smoke(stage_args("live", engine="transformers"), make=QwenAws) == 0
    recs = [a for a in aws_cli.attempts("qwen") if a.get("kind") == "smoke-hf"]
    assert [(a["instance_type"], a["status"]) for a in recs] == [("ml.g6e.12xlarge", "Failed"), ("ml.g5.12xlarge", "Succeeded")]
    assert recs[-1]["substitution"] and recs[-1]["checkpoints"].startswith("s3://")
    proj = json.loads((config.results_dir("live") / "qwen_projection.json").read_text())
    n_items = sum(len(prompts.all_items(d)) for d in dss)
    assert proj["complete"] and proj["transformers"]["projected_seconds"] >= 2.0 * n_items
    assert proj["projected_usd"] == pytest.approx(proj["vllm"]["projected_usd"] + proj["transformers"]["projected_usd"])
    monkeypatch.setitem(config.load("budgets"), "aws_spend_limit_usd", proj["projected_usd"] / 2)
    with pytest.raises(RuntimeError, match="llm run refused"):
        cli.run_live(stage_args("live", thinking="subsample"), make=QwenAws)
    with pytest.raises(RuntimeError, match="run `llm project-thinking` first"):
        cli.run_live(stage_args("live", thinking="all"), make=QwenAws)


def test_vllm_prompt_method_sums_the_option_positions(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace as NS

    class FakeLLM:
        def generate(self, reqs, sp, use_tqdm=False):
            assert sp.prompt_logprobs == 0
            return [NS(prompt_logprobs=[None] + [{t: NS(logprob=-0.1 * (j + 1))} for j, t in enumerate(r.prompt_token_ids[1:])])
                    for r in reqs]

    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kw: NS(**kw)
    fake.TokensPrompt = lambda prompt_token_ids: NS(prompt_token_ids=prompt_token_ids)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    eng = qwen_infer.VllmEngine.__new__(qwen_infer.VllmEngine)
    eng.llm, eng.requests, eng.method = FakeLLM(), 0, "prompt"
    got = eng.score([[9, 9, 9]], [[5, 1], [6, 7, 1]])
    assert np.allclose(got, [[-(0.3 + 0.4), -(0.3 + 0.4 + 0.5)]]) and eng.requests == 2


def test_fetch_checks_identifiers_against_jev(approved, monkeypatch):
    from conftest import stage_args
    from ciayn import results
    from ciayn.baseline import cli as aws_cli
    from ciayn.data import datasets
    from ciayn.llm import cli
    sst2 = datasets.load("sst2")
    QwenAws.files = canned_run([sst2], "full-hf")
    aws_cli.record_attempt({"model": "qwen", "kind": "full-hf", "status": "Succeeded", "region": "us-east-2",
                            "instance_type": "ml.g6e.12xlarge", "execution_arn": "arn:x", "billable_seconds": 100.0,
                            "cost_usd": 0.4, "checkpoints": "s3://b/checkpoints/qwen/r/", "substitution": "s", "steps": {}})
    cli.fetch(stage_args("live", "sst2"), make=QwenAws)
    df = results.read_predictions("live", "sst2", "qwen")
    assert sorted(df.id) == sorted(sst2.jev_ids) and df.valid.all()
    think = results.read_predictions("live", "sst2", "qwen_think")
    assert sorted(think.id) == sorted(thinking_subsample(sst2))
    rows = [json.loads(x) for x in QwenAws.files["sst2.nothink.jsonl"].decode().splitlines()][1:]
    with pytest.raises(RuntimeError, match="do not match the expected identifiers"):
        cli.ingest(sst2, rows, "qwen", list(sst2.jev_ids), {})


# ---- the vLLM scoring path, against a fake engine ------------------------------------------------

def test_trie_edges_are_shared_between_options_and_ordered_by_depth():
    edges = qwen_infer.trie_edges([[5, 6, 1], [5, 7, 1], [8, 1]])
    assert edges[0][0] == () and {e for e in edges if e[0] == ()} == {((), 5), ((), 8)}
    assert ((5,), 6) in edges and ((5,), 7) in edges and len(edges) == 7, "the shared first token is asked for once"


def test_vllm_scoring_sums_raw_logprobs_over_trie_edges(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace as NS

    def logprob(context, tok):                      # a deterministic stand-in for the model
        return -((sum(context) * 31 + tok * 7) % 13) / 10 - 0.05

    calls = []

    class FakeLLM:
        def generate(self, reqs, sps, use_tqdm=False):
            calls.append(len(reqs))
            out = []
            for r, sp in zip(reqs, sps):
                assert sp.max_tokens == 1 and len(sp.allowed_token_ids) == 1 and sp.prompt_logprobs is None
                tok = sp.allowed_token_ids[0]
                out.append(NS(outputs=[NS(token_ids=[tok], logprobs=[{tok: NS(logprob=logprob(r.prompt_token_ids, tok))}])]))
            return out

    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kw: NS(prompt_logprobs=None, **kw)
    fake.TokensPrompt = lambda prompt_token_ids: NS(prompt_token_ids=prompt_token_ids)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    eng = qwen_infer.VllmEngine.__new__(qwen_infer.VllmEngine)
    eng.llm, eng.requests, eng.method = FakeLLM(), 0, "edges"
    prefixes, options = [[100, 101], [200]], [[5, 6, 1], [5, 7, 1], [8, 1]]
    got = eng.score(prefixes, options)
    want = [[sum(logprob(p + o[:t], o[t]) for t in range(len(o))) for o in options] for p in prefixes]
    assert np.allclose(got, want)
    assert calls == [2, 2 * 6], "one request per item first (it fills the prefix cache), then the other edges"
    assert eng.requests == 2 * 7, "7 trie edges per item instead of 8 option tokens"


def test_reasoning_generated_in_one_job_is_scored_in_another(tmp_path):
    """`generate` saves the reasoning tokens; `given` scores after them and equals scoring in one go."""
    model, tok = tiny_qwen(None)
    items = [{"id": f"x{i}", "messages": [{"role": "system", "content": "Options: - good - bad Answer"},
                                         {"role": "user", "content": w}]} for i, w in enumerate(["good movie", "bad movie"])]
    base = {"datasets": ["toy"], "options": {"toy": ["good", "bad"]}, "example_id": {"toy": "x0"}, "dtype": "float32",
            "max_thinking_tokens": 3, "items_per_call": 1, "think_items_per_call": 1}
    eng = qwen_infer.HfEngine("unused", base, model=model, tok=tok)

    def run(name, mode, rows, nothink=()):
        src = tmp_path / name
        src.mkdir()
        (src / "toy.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        (src / "job.json").write_text(json.dumps({**base, "think_mode": {"toy": mode},
                                                  "sets": {"toy": {"nothink": list(nothink), "think": ["x0", "x1"], "latency": []}}}))
        qwen_infer.main(src, "unused", tmp_path / (name + "-out"), engine=eng)
        return tmp_path / (name + "-out")

    gen = run("gen", "generate", items)
    saved = {r["id"]: r for r in map(json.loads, (gen / "toy.think_reasoning.jsonl").read_text().splitlines())}
    assert not (gen / "toy.think.jsonl").exists() and all("p" not in r and r["reasoning_ids"] for r in saved.values())
    given = run("given", "given", [{**it, "reasoning_ids": saved[it["id"]]["reasoning_ids"],
                                    "thinking_truncated": saved[it["id"]]["thinking_truncated"]} for it in items])
    one = run("one", "score", items)
    a = [json.loads(x)["p"] for x in (given / "toy.think.jsonl").read_text().splitlines()]
    b = [json.loads(x)["p"] for x in (one / "toy.think.jsonl").read_text().splitlines()]
    assert np.allclose(a, b, atol=1e-5)
    assert eng.cache_check["max_abs_logprob_difference"] < 1e-3, "cached scoring was checked against full recomputation"


def test_cache_check_refuses_a_wrong_cache(monkeypatch):
    model, tok = tiny_qwen(None)
    eng = qwen_infer.HfEngine("unused", {"dtype": "float32"}, model=model, tok=tok)
    prefix = tok.encode("<|im_start|> user good movie <|im_end|> <|im_start|> assistant", add_special_tokens=False)
    opts = qwen_infer.option_ids(tok, ["good", "bad", "card_arrival"])
    assert eng.check(prefix, opts)["max_abs_logprob_difference"] < 1e-3
    monkeypatch.setattr(eng, "score", lambda p, o: np.zeros((1, len(o))))
    with pytest.raises(RuntimeError, match="differs from full recomputation"):
        eng.check(prefix, opts)


def test_shared_prefix_gives_the_same_scores(tmp_path):
    model, tok = tiny_qwen(None)
    job = {"dtype": "float32", "option_chunk": 2}
    plain = qwen_infer.HfEngine("unused", job, model=model, tok=tok)
    shared = qwen_infer.HfEngine("unused", job, model=model, tok=tok)
    system = "<|im_start|> system Options: - good - bad Answer the the the the the the the the the the the the the the the " * 3
    prefixes = [tok.encode(system + f"<|im_end|> <|im_start|> user {w} <|im_end|> <|im_start|> assistant", add_special_tokens=False)
                for w in ("good movie", "bad movie yes", "card")]
    common = qwen_infer.common_prefix(prefixes)
    assert len(common) >= 32 and all(p[: len(common)] == common for p in prefixes)
    shared.set_shared(common[:-1])
    opts = qwen_infer.option_ids(tok, ["good", "bad", "card_arrival"])
    assert np.allclose(plain.score(prefixes, opts), shared.score(prefixes, opts), atol=1e-4)
    other = tok.encode("<|im_start|> user good <|im_end|> <|im_start|> assistant", add_special_tokens=False)
    assert np.allclose(plain.score([other], opts), shared.score([other], opts), atol=1e-4), "a prompt without the shared prefix is computed in full"


def test_equal_length_items_are_batched_and_score_the_same_as_one_by_one():
    model, tok = tiny_qwen(None)
    base = {"dtype": "float32", "option_chunk": 2}
    one = qwen_infer.HfEngine("unused", {**base, "max_items_per_pass": 1}, model=model, tok=tok)
    many = qwen_infer.HfEngine("unused", {**base, "max_items_per_pass": 8}, model=model, tok=tok)
    system = "<|im_start|> system Options: - good - bad Answer " + "the " * 40
    texts = ["good movie", "bad movie", "card yes", "good", "the good bad movie", "no no"]         # lengths 2, 2, 2, 1, 4, 2
    prefixes = [tok.encode(system + f"<|im_end|> <|im_start|> user {w} <|im_end|> <|im_start|> assistant", add_special_tokens=False)
                for w in texts]
    for e in (one, many):
        e.set_shared(qwen_infer.common_prefix(prefixes)[:-1])
    calls = []
    real = many._states
    many._states = lambda group: calls.append(len(group)) or real(group)
    opts = qwen_infer.option_ids(tok, ["good", "bad", "card_arrival"])
    assert np.allclose(one.score(prefixes, opts), many.score(prefixes, opts), atol=1e-4)
    assert sorted(calls) == [1, 1, 4], "four items of equal length in one pass; the others alone"


def test_a_smoke_run_with_failed_items_gives_no_projection(approved):
    from conftest import stage_args
    from ciayn.baseline import cli as aws_cli
    from ciayn.llm import cli
    run = {"engine": "transformers", "load_s": 20.0, "datasets": {"agnews": {"nothink": {"items": 8, "seconds": 2.0, "invalid": 8},
                                                                                "latency_s": [1.2]}}}
    QwenAws.files = {"run.json": json.dumps(run).encode()}
    aws_cli.record_attempt({"model": "qwen", "kind": "smoke-hf", "status": "Succeeded", "region": "us-east-2", "dataset": "all",
                            "instance_type": "ml.g6.12xlarge", "execution_arn": "arn:s", "billable_seconds": 400.0, "cost_usd": 0.6,
                            "checkpoints": "s3://b/checkpoints/qwen/s/", "steps": {"Infer": {"billable_seconds": 400.0}}})
    with pytest.raises(RuntimeError, match="no projection is made from a run with failures"):
        cli.smoke(stage_args("live", engine="transformers"), make=QwenAws)
    assert not (config.results_dir("live") / "qwen_projection.json").exists()
