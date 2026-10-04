"""Copy the response cache, the ledger and the exchange log outside the working copy (design D5).

Runs after every live stage. The destination is the project bucket once it exists (recorded
by `baseline deploy` in results/live/aws_stack.json), and a local directory next to the
project until then.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources() -> list[Path]:
    files = [config.CACHE / "responses.sqlite"] + sorted((config.CACHE / "exchanges").glob("*.jsonl"))
    return [f for f in files if f.exists()]


def record_path() -> Path:
    return config.results_dir("live") / "backup.json"


def stack_record() -> dict | None:
    p = config.RESULTS / "live" / "aws_stack.json"
    return json.loads(p.read_text()) if p.exists() else None


def run(args=None) -> dict:
    files = sources()
    stack = stack_record()
    entries = [{"name": str(f.relative_to(config.CACHE)), "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in files]
    if stack and stack.get("bucket"):
        import boto3

        s3 = boto3.client("s3", region_name=stack["region"])
        for f, e in zip(files, entries):
            s3.upload_file(str(f), stack["bucket"], f"backup/{e['name']}", ExtraArgs={"ChecksumAlgorithm": "SHA256"})
            head = s3.head_object(Bucket=stack["bucket"], Key=f"backup/{e['name']}")
            if head["ContentLength"] != e["bytes"]:
                raise RuntimeError(f"backup of {e['name']} is incomplete in s3://{stack['bucket']}/backup/")
        location = f"s3://{stack['bucket']}/backup/"
    else:
        dest = config.BACKUP
        for f, e in zip(files, entries):
            target = dest / e["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            if sha256_file(target) != e["sha256"]:
                raise RuntimeError(f"backup copy of {e['name']} differs from the original")
        location = str(dest)
    rec = {"location": location, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": entries}
    record_path().write_text(json.dumps(rec, indent=1))
    print(f"backup: {len(entries)} files ({sum(e['bytes'] for e in entries):,} bytes) copied to {location} at {rec['time']}")
    return rec


def local_matches_latest() -> tuple[bool, str]:
    """Whether the local cache, ledger and exchange log are exactly what the latest backup holds."""
    if not record_path().exists():
        return False, "no backup has been recorded"
    rec = json.loads(record_path().read_text())
    current = {str(f.relative_to(config.CACHE)): sha256_file(f) for f in sources()}
    recorded = {e["name"]: e["sha256"] for e in rec["files"]}
    if current != recorded:
        return False, "the local cache, ledger or exchange log changed since the latest backup, or a file is missing"
    return True, rec["location"]
