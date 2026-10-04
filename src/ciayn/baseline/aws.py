"""Thin wrapper over the AWS calls the baseline needs. Tests substitute a fake with the same methods."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Aws:
    def __init__(self, region: str):
        import boto3

        self.region = region
        s = boto3.session.Session(region_name=region)
        self.cfn, self.s3, self.sm, self.sq = (s.client(n) for n in ("cloudformation", "s3", "sagemaker", "service-quotas"))

    # ---- CloudFormation ------------------------------------------------------------------------

    def stack_outputs(self, name: str) -> dict | None:
        try:
            st = self.cfn.describe_stacks(StackName=name)["Stacks"][0]
        except self.cfn.exceptions.ClientError:
            return None
        if st["StackStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"):
            return None
        return {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}

    def deploy_stack(self, name: str, template_body: str, operator_role: str) -> dict:
        self.cfn.create_stack(StackName=name, TemplateBody=template_body, Capabilities=["CAPABILITY_NAMED_IAM"],
                              Parameters=[{"ParameterKey": "OperatorRoleName", "ParameterValue": operator_role}],
                              Tags=[{"Key": "project", "Value": "ciayn"}])
        self.cfn.get_waiter("stack_create_complete").wait(StackName=name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
        return self.stack_outputs(name)

    def update_stack(self, name: str, template_body: str, operator_role: str) -> dict:
        try:
            self.cfn.update_stack(StackName=name, TemplateBody=template_body, Capabilities=["CAPABILITY_NAMED_IAM"],
                                  Parameters=[{"ParameterKey": "OperatorRoleName", "ParameterValue": operator_role}])
        except self.cfn.exceptions.ClientError as e:
            if "No updates are to be performed" not in str(e):
                raise
            return self.stack_outputs(name)
        self.cfn.get_waiter("stack_update_complete").wait(StackName=name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
        return self.stack_outputs(name)

    def delete_stack(self, name: str) -> None:
        self.cfn.delete_stack(StackName=name)
        self.cfn.get_waiter("stack_delete_complete").wait(StackName=name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})

    # ---- S3 ------------------------------------------------------------------------------------

    def upload_dir(self, local: Path, bucket: str, prefix: str) -> int:
        n = 0
        for f in sorted(p for p in Path(local).rglob("*") if p.is_file()):
            self.s3.upload_file(str(f), bucket, f"{prefix}/{f.relative_to(local).as_posix()}")
            n += 1
        return n

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        keys = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        return keys

    def download(self, bucket: str, key: str, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(bucket, key, str(path))

    def empty_bucket(self, bucket: str) -> int:
        n = 0
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self.s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
                n += len(keys)
        return n

    # ---- SageMaker -----------------------------------------------------------------------------

    def quota(self, instance_type: str, kind: str) -> float | None:
        """Account quota for `<instance> for <processing|training> job usage`, or None when it cannot be read."""
        try:
            for page in self.sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
                for q in page["Quotas"]:
                    if q["QuotaName"] == f"{instance_type} for {kind} job usage":
                        return float(q["Value"])
        except Exception:
            return None
        return None

    def upsert_pipeline(self, name: str, definition: dict, role_arn: str) -> None:
        body = json.dumps(definition, sort_keys=True)
        try:
            self.sm.create_pipeline(PipelineName=name, PipelineDefinition=body, RoleArn=role_arn,
                                    Tags=[{"Key": "project", "Value": "ciayn"}])
        except self.sm.exceptions.ClientError as e:
            if "already exists" not in str(e):
                raise
            self.sm.update_pipeline(PipelineName=name, PipelineDefinition=body, RoleArn=role_arn)

    def delete_pipeline(self, name: str) -> None:
        try:
            self.sm.delete_pipeline(PipelineName=name)
        except self.sm.exceptions.ClientError:
            pass

    def start(self, name: str, params: dict) -> str:
        r = self.sm.start_pipeline_execution(
            PipelineName=name, PipelineParameters=[{"Name": k, "Value": str(v)} for k, v in params.items()])
        return r["PipelineExecutionArn"]

    def waiting_for_capacity(self, arn: str) -> str | None:
        """The name of a training job of this execution that is still waiting for an instance, if any."""
        for s in self.sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]:
            tj = s.get("Metadata", {}).get("TrainingJob", {}).get("Arn")
            if tj and s["StepStatus"] == "Executing":
                j = self.sm.describe_training_job(TrainingJobName=tj.split("/")[-1])
                msg = (j.get("SecondaryStatusTransitions") or [{}])[-1].get("StatusMessage", "")
                if j.get("SecondaryStatus") == "Pending" and "capacity" in msg.lower():
                    return tj.split("/")[-1]
        return None

    def wait(self, arn: str, poll: int = 30, on_poll=None, capacity_timeout_s: float | None = None) -> dict:
        """Block until the execution ends; returns its status, failure reason and per-step job details.

        With `capacity_timeout_s`, an execution whose training job has waited that long for an instance
        is stopped and reported as a capacity failure, so the caller can try another instance type.
        """
        pending_since, no_capacity = None, None
        while True:
            d = self.sm.describe_pipeline_execution(PipelineExecutionArn=arn)
            status = d["PipelineExecutionStatus"]
            if on_poll:
                on_poll(status)
            if status not in ("Executing", "Stopping"):
                break
            if capacity_timeout_s and status == "Executing":
                job = self.waiting_for_capacity(arn)
                pending_since = (pending_since or time.time()) if job else None
                if job and time.time() - pending_since > capacity_timeout_s:
                    no_capacity = f"CapacityError: {job} waited {capacity_timeout_s:.0f}s for an instance; the execution was stopped"
                    self.sm.stop_pipeline_execution(PipelineExecutionArn=arn)
                    pending_since = None
                    capacity_timeout_s = None
            time.sleep(poll)
        steps = {}
        for s in self.sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]:
            info = {"status": s["StepStatus"], "failure": s.get("FailureReason")}
            meta = s.get("Metadata", {})
            if "ProcessingJob" in meta:
                job = meta["ProcessingJob"]["Arn"].split("/")[-1]
                j = self.sm.describe_processing_job(ProcessingJobName=job)
                start, end = j.get("ProcessingStartTime"), j.get("ProcessingEndTime")
                info.update(job=job, image=j["AppSpecification"]["ImageUri"], failure=info["failure"] or j.get("FailureReason"),
                            billable_seconds=(end - start).total_seconds() if start and end else 0.0)
            if "TrainingJob" in meta:
                job = meta["TrainingJob"]["Arn"].split("/")[-1]
                j = self.sm.describe_training_job(TrainingJobName=job)
                info.update(job=job, image=j["AlgorithmSpecification"]["TrainingImage"],
                            failure=info["failure"] or j.get("FailureReason"),
                            billable_seconds=float(j.get("BillableTimeInSeconds", 0)),
                            output=f"{j['OutputDataConfig']['S3OutputPath'].rstrip('/')}/{job}/output/output.tar.gz")
            steps[s["StepName"]] = info
        return {"status": status, "failure": no_capacity or d.get("FailureReason"), "steps": steps}
