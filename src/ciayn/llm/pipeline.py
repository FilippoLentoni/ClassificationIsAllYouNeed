"""The `ciayn-qwen` SageMaker pipeline, built as plain data (design D3).

1. `Stage`: a CPU processing job that copies the pinned weights from Hugging Face to S3 once
   (it ends at once when the manifest for the revision exists).
2. `IfInfer` -> `Infer`: when the `RunInference` parameter is "yes", a GPU *training* job used for
   inference, in the AWS vLLM container. Every GPU processing-job quota in the account is 0,
   while GPU training-job quotas exist; the substitution is recorded with every run.
Nothing here talks to AWS.
"""
from __future__ import annotations

from .. import config
from ..baseline import pipeline as base

SUBSTITUTION = ("inference ran in a SageMaker training job, because the account's quota for GPU "
                "processing jobs is 0 in every region checked, while GPU training-job quotas exist")


def stage_image(region: str) -> str:
    return f"{config.load('baseline')['registries'][region]}.dkr.ecr.{region}.amazonaws.com/{config.load('llm')['stage']['image']}"


def infer_image(region: str) -> str:
    cfg = config.load("llm")["infer"]
    return f"{cfg['registry']}.dkr.ecr.{region}.amazonaws.com/{cfg['image']}"


def weights_uri(bucket: str) -> str:
    cfg = config.load("llm")
    return f"s3://{bucket}/{cfg['weights_prefix']}/{cfg['revision']}/"


def definition(region: str, bucket: str, role_arn: str, max_runtime_s: int | None = None) -> dict:
    cfg = config.load("llm")
    max_runtime_s = int(max_runtime_s or cfg["infer"]["max_runtime_s"])
    root = f"s3://{bucket}"
    stage = {
        "Name": "Stage", "Type": "Processing",
        "Arguments": {
            "ProcessingResources": {"ClusterConfig": {"InstanceType": cfg["stage"]["instance_type"], "InstanceCount": 1,
                                                      "VolumeSizeInGB": cfg["stage"]["volume_gb"]}},
            "AppSpecification": {"ImageUri": stage_image(region),
                                 "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/stage_weights.py",
                                                         cfg["model_id"], cfg["revision"], bucket, cfg["weights_prefix"]]},
            "RoleArn": role_arn,
            "ProcessingInputs": [base._s3_input("code", f"{root}/code-qwen", "/opt/ml/processing/input/code")],
            "StoppingCondition": {"MaxRuntimeInSeconds": cfg["stage"]["max_runtime_s"]},
        },
    }
    infer = {
        "Name": "Infer", "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": infer_image(region), "TrainingInputMode": "File",
                                       "ContainerEntrypoint": ["python3", "/opt/ml/input/data/code/qwen_infer.py"]},
            "RoleArn": role_arn,
            "InputDataConfig": [base._channel("prompts", base._join(f"{root}/input-qwen", base._param("RunId"))),
                                base._channel("weights", weights_uri(bucket)),
                                base._channel("code", f"{root}/code-qwen")],
            "OutputDataConfig": {"S3OutputPath": base._join(f"{root}/output", "qwen", base._param("RunId"))},
            "CheckpointConfig": {"S3Uri": base._join(f"{root}/checkpoints", "qwen", base._param("RunId")),
                                 "LocalPath": "/opt/ml/checkpoints"},
            "ResourceConfig": {"InstanceType": base._param("InstanceType"), "InstanceCount": 1,
                               "VolumeSizeInGB": cfg["infer"]["volume_gb"]},
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_s},
            "Environment": {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "VLLM_NO_USAGE_STATS": "1",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        },
    }
    cond = {"Name": "IfInfer", "Type": "Condition", "DependsOn": ["Stage"],
            "Arguments": {"Conditions": [{"Type": "Equals", "LeftValue": base._param("RunInference"), "RightValue": "yes"}],
                          "IfSteps": [infer], "ElseSteps": []}}
    return {"Version": "2020-12-01",
            "Parameters": [{"Name": "RunId", "Type": "String"},
                           {"Name": "InstanceType", "Type": "String", "DefaultValue": cfg["infer"]["instance_types"][0]},
                           {"Name": "RunInference", "Type": "String", "DefaultValue": "yes"}],
            "Steps": [stage, cond]}


def worst_case_cost(instance_type: str, region: str, max_runtime_s: int, inference: bool = True) -> float:
    """Upper bound in USD for one execution: both jobs running to their stopping conditions."""
    cfg = config.load("llm")
    total = base.cost(cfg["stage"]["instance_type"], region, cfg["stage"]["max_runtime_s"])
    return total + (base.cost(instance_type, region, max_runtime_s) if inference else 0.0)
