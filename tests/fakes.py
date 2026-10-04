import json

import httpx2
import numpy as np

from ciayn.backends.base import Backend, Prediction

SECRET = "apikey_test_0123456789abcdefSECRETVALUE"


class CountingBackend(Backend):
    """Deterministic fake remote backend that counts calls."""

    name = "fake"
    remote = True

    def __init__(self, batch: int = 1):
        self.calls = 0
        self.inputs = 0
        self.batch = batch

    @property
    def model_version(self):
        return "fake-1"

    def calls_needed(self, n):
        return -(-n // self.batch)

    def _predict(self, ds, texts):
        self.calls += self.calls_needed(len(texts))
        self.inputs += len(texts)
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 2**32)
            out.append(Prediction.make(rng.random(len(ds.classes)) + 0.01, meta={"latency_s": 0.0}))
        return out


class FakeJevServer:
    """Stands in for api.typesafe.ai underneath the real SDK client: nothing leaves the machine.

    Probabilities depend on the input and follow the order of the criteria in the request, so
    tests can check both the mapping and that the request carried the options in class order.
    """

    def __init__(self, model="jev-1.13.0", missing_class=False, fail_on=()):
        self.model, self.missing_class, self.fail_on = model, missing_class, set(fail_on)
        self.calls = 0
        self.requests: list[dict] = []

    @staticmethod
    def probabilities(state: str, labels: list[str]) -> dict:
        raw = np.linspace(1, 2, len(labels)) + (len(state) % 7)
        return {lab: float(v) for lab, v in zip(labels, raw / raw.sum())}

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        body = json.loads(request.content)
        self.requests.append(body)
        if body["state"] in self.fail_on:
            raise httpx2.ConnectError(f"boom for key {SECRET}")
        (name, q), = body["questions"].items()
        probs = self.probabilities(body["state"], list(q["criteria"]))
        if self.missing_class:
            probs.pop(list(q["criteria"])[-1])
        answer = {"type": "choice", "choice": max(probs, key=probs.get), "confidence": 0.5,
                  "probabilities": dict(reversed(probs.items()))}   # reversed: the mapping must not rely on order
        return httpx2.Response(200, json={"model": self.model, "usage": {"input_tokens": 100, "output_tokens": 0},
                                          "answers": {name: answer}})

    def transport(self):
        return httpx2.MockTransport(self.handler)


class FakeAws:
    """Scripted SageMaker: `outcomes` maps (region, instance type) to a failure reason; anything else succeeds."""
    log: list = []
    outcomes: dict = {}
    quotas: dict = {}

    def __init__(self, region):
        self.region = region

    def stack_outputs(self, name):
        return {"BucketName": f"bucket-{self.region}", "ExecutionRoleArn": f"arn:role-{self.region}"}

    def quota(self, itype, kind):
        return self.quotas.get((self.region, itype))

    def upload_dir(self, *a):
        return 0

    def upsert_pipeline(self, *a):
        pass

    def start(self, name, params):
        FakeAws.log.append(("start", self.region, params["InstanceType"], params.get("Dataset", "all")))
        self.itype = params["InstanceType"]
        return f"arn:exec/{self.region}-{self.itype}-{params.get('Dataset', 'all')}-{params.get('RunId', '')}"

    def wait(self, arn, **kw):
        reason = self.outcomes.get((self.region, self.itype))
        if reason:
            return {"status": "Failed", "failure": None, "steps": {"Preprocess": {"status": "Failed", "failure": reason}}}
        return {"status": "Succeeded", "failure": None,
                "steps": {"Preprocess": {"status": "Succeeded", "failure": None, "billable_seconds": 60.0},
                          "Train": {"status": "Succeeded", "failure": None, "billable_seconds": 120.0, "output": "s3://b/k"}}}

    def delete_pipeline(self, *a): FakeAws.log.append(("delete_pipeline", self.region))
    def empty_bucket(self, *a): return 0
    def delete_stack(self, *a): FakeAws.log.append(("delete_stack", self.region))
