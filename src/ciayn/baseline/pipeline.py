"""The SageMaker pipeline definition, built as plain data (design D6).

Two steps per execution, parameterised by dataset and instance type:

1. `Preprocess`, a processing job in the scikit-learn container running `preprocess.py`;
2. `Train`, a training job in the XGBoost container running `train.py` on the features the
   first step produced.

Nothing here talks to AWS, so the definition can be generated and compared offline.
"""
from __future__ import annotations

from .. import config


def image_uri(region: str, which: str) -> str:
    cfg = config.load("baseline")
    return f"{cfg['registries'][region]}.dkr.ecr.{region}.amazonaws.com/{cfg['images'][which]}"


def _param(name: str) -> dict:
    return {"Get": f"Parameters.{name}"}


def _join(*values) -> dict:
    return {"Std:Join": {"On": "/", "Values": list(values)}}


def _s3_input(name: str, uri, local: str) -> dict:
    return {"InputName": name, "AppManaged": False,
            "S3Input": {"S3Uri": uri, "LocalPath": local, "S3DataType": "S3Prefix", "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated", "S3CompressionType": "None"}}


def _channel(name: str, uri) -> dict:
    return {"ChannelName": name, "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri,
                                                                 "S3DataDistributionType": "FullyReplicated"}}}


def definition(region: str, bucket: str, role_arn: str) -> dict:
    cfg = config.load("baseline")
    base = f"s3://{bucket}"
    resources = {"InstanceType": _param("InstanceType"), "InstanceCount": 1, "VolumeSizeInGB": cfg["volume_gb"]}
    preprocess = {
        "Name": "Preprocess", "Type": "Processing",
        "Arguments": {
            "ProcessingResources": {"ClusterConfig": dict(resources)},
            "AppSpecification": {"ImageUri": image_uri(region, "preprocess"),
                                 "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/preprocess.py"]},
            "RoleArn": role_arn,
            "ProcessingInputs": [
                _s3_input("data", _join(f"{base}/input", _param("Dataset")), "/opt/ml/processing/input/data"),
                _s3_input("code", f"{base}/code", "/opt/ml/processing/input/code")],
            "ProcessingOutputConfig": {"Outputs": [{
                "OutputName": "features", "AppManaged": False,
                "S3Output": {"S3Uri": _join(f"{base}/output", _param("Dataset"), _param("RunId"), "features"),
                             "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]},
            "StoppingCondition": {"MaxRuntimeInSeconds": cfg["max_runtime_s"]["preprocess"]},
        },
    }
    train = {
        "Name": "Train", "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": image_uri(region, "train"), "TrainingInputMode": "File",
                                       "ContainerEntrypoint": ["python3", "/opt/ml/input/data/code/train.py"]},
            "RoleArn": role_arn,
            "InputDataConfig": [
                _channel("features", {"Get": "Steps.Preprocess.ProcessingOutputConfig.Outputs['features'].S3Output.S3Uri"}),
                _channel("code", f"{base}/code")],
            "OutputDataConfig": {"S3OutputPath": _join(f"{base}/output", _param("Dataset"), _param("RunId"), "train")},
            "ResourceConfig": dict(resources),
            "StoppingCondition": {"MaxRuntimeInSeconds": cfg["max_runtime_s"]["train"]},
        },
    }
    return {"Version": "2020-12-01",
            "Parameters": [{"Name": "Dataset", "Type": "String"},
                           {"Name": "RunId", "Type": "String"},
                           {"Name": "InstanceType", "Type": "String", "DefaultValue": cfg["instance_types"][0]}],
            "Steps": [preprocess, train]}


def worst_case_cost(instance_type: str, region: str) -> float:
    """Upper bound in USD for one execution: both jobs running to their stopping condition."""
    cfg = config.load("baseline")
    hours = (cfg["max_runtime_s"]["preprocess"] + cfg["max_runtime_s"]["train"]) / 3600
    return hours * cfg["price_usd_per_hour"][instance_type] * cfg["region_price_factor"][region]


def cost(instance_type: str, region: str, billable_seconds: float) -> float:
    cfg = config.load("baseline")
    return billable_seconds / 3600 * cfg["price_usd_per_hour"][instance_type] * cfg["region_price_factor"][region]
