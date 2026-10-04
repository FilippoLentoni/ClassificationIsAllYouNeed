"""Configuration for the BERT and Qwen comparisons (change add-bert-qwen-ensemble)."""
from ciayn import config


def test_new_model_configs_load_and_are_pinned():
    enc, llm = config.load("encoder"), config.load("llm")
    assert len(enc["revision"]) == 40 and len(llm["revision"]) == 40, "model revisions are pinned commit hashes"
    assert enc["pipeline_name"] == "ciayn-bert" and llm["pipeline_name"] == "ciayn-qwen"
    assert set(enc["max_length"]) == set(config.DATASETS)
    assert llm["infer"]["instance_types"][0] == "ml.g6e.12xlarge"


def test_spend_limit_and_gpu_prices():
    assert config.load("budgets")["aws_spend_limit_usd"] == 75.0
    prices = config.load("baseline")["price_usd_per_hour"]
    for itype in config.load("encoder")["instance_types"] + config.load("llm")["infer"]["instance_types"]:
        assert prices[itype] > 1.0, itype
    assert config.load("llm")["stage"]["instance_type"] in prices
