"""The `ciayn-bert` SageMaker pipeline, built as plain data (design D2).

One training step per execution, parameterised by dataset and instance type, in the AWS Hugging
Face PyTorch training container running `bert_train.py`. Nothing here talks to AWS.
"""
from __future__ import annotations

from .. import config
from ..baseline import pipeline as base


def image_uri(region: str) -> str:
    cfg = config.load("encoder")
    return f"{cfg['registry']}.dkr.ecr.{region}.amazonaws.com/{cfg['image']}"


def definition(region: str, bucket: str, role_arn: str) -> dict:
    cfg = config.load("encoder")
    root = f"s3://{bucket}"
    train = {
        "Name": "Train", "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": image_uri(region), "TrainingInputMode": "File",
                                       "ContainerEntrypoint": ["python3", "/opt/ml/input/data/code/bert_train.py"]},
            "RoleArn": role_arn,
            "InputDataConfig": [base._channel("data", base._join(f"{root}/input-bert", base._param("Dataset"))),
                                base._channel("code", f"{root}/code-bert")],
            "OutputDataConfig": {"S3OutputPath": base._join(f"{root}/output", base._param("Dataset"), base._param("RunId"), "bert")},
            "ResourceConfig": {"InstanceType": base._param("InstanceType"), "InstanceCount": 1, "VolumeSizeInGB": cfg["volume_gb"]},
            "StoppingCondition": {"MaxRuntimeInSeconds": cfg["max_runtime_s"]},
            "Environment": {"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"},
        },
    }
    return {"Version": "2020-12-01",
            "Parameters": [{"Name": "Dataset", "Type": "String"}, {"Name": "RunId", "Type": "String"},
                           {"Name": "InstanceType", "Type": "String", "DefaultValue": cfg["instance_types"][0]}],
            "Steps": [train]}


def worst_case_cost(instance_type: str, region: str) -> float:
    """Upper bound in USD for one execution: the training job running to its stopping condition."""
    return base.cost(instance_type, region, config.load("encoder")["max_runtime_s"])
