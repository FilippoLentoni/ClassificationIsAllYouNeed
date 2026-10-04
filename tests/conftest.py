import os
from types import SimpleNamespace

import pytest

# The suite never makes network calls to model APIs or AWS; make sure no credentials leak in.
os.environ.pop("TYPESAFE_API_KEY", None)
os.environ["CIAYN_SKIP_BACKUP"] = "1"   # tests never copy the real cache


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Every test gets its own Jev ledger, so tests never touch the real one."""
    from ciayn import config
    monkeypatch.setattr(config, "LEDGER", tmp_path / "ledger.sqlite")
    return config.LEDGER


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Results, response cache and exchange log redirected to a temporary directory."""
    from ciayn import config
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "BACKUP", tmp_path / "backup")
    monkeypatch.setattr(config, "PAPER", tmp_path / "paper")
    return tmp_path


@pytest.fixture
def live(workspace, monkeypatch):
    """Live mode against a fake Jev server underneath the real SDK client."""
    from ciayn.backends import jev
    from fakes import SECRET, FakeJevServer
    server = FakeJevServer()
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    monkeypatch.setattr(jev, "TRANSPORT_FACTORY", server.transport)
    return SimpleNamespace(server=server, root=workspace, secret=SECRET)


def stage_args(mode="mock", dataset="all", **kw):
    import argparse
    kw.setdefault("rerun", False)
    kw.setdefault("dry_run", False)
    kw.setdefault("no_build", True)
    return argparse.Namespace(mode=mode, dataset=dataset, refreeze=False, **kw)


@pytest.fixture(scope="session")
def german():
    from ciayn.data import datasets
    return datasets.load("german")


@pytest.fixture(scope="session")
def wdbc():
    from ciayn.data import datasets
    return datasets.load("wdbc")


@pytest.fixture(scope="session")
def sst2():
    from ciayn.data import datasets
    return datasets.load("sst2")


@pytest.fixture(scope="session")
def _mock_predictions(tmp_path_factory):
    """Mock predictions for the three small datasets, generated once per test session."""
    from ciayn import config, predict
    root = tmp_path_factory.mktemp("mock_predictions")
    old = config.RESULTS
    config.RESULTS = root
    try:
        for name in ("german", "wdbc", "sst2"):
            predict.run(stage_args("mock", name))
    finally:
        config.RESULTS = old
    return root


@pytest.fixture
def mock_run(workspace, _mock_predictions, monkeypatch):
    """A workspace that already holds mock predictions for german, wdbc and sst2; fast settings."""
    import shutil
    from ciayn import config
    shutil.copytree(_mock_predictions, workspace / "results", dirs_exist_ok=True)
    monkeypatch.setitem(config.load("metrics"), "bootstrap", 40)
    return workspace


@pytest.fixture
def approved(workspace):
    """The AWS template counts as approved; SageMaker is the scripted FakeAws."""
    import json
    from ciayn import config
    from fakes import FakeAws
    FakeAws.log, FakeAws.outcomes, FakeAws.quotas = [], {}, {}
    (config.results_dir("live") / "aws_stack.json").write_text(json.dumps({"approved_at": "2026-09-30T00:00:00Z"}))
    return workspace
