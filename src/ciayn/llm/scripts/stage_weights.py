"""SageMaker processing job (CPU): copy a pinned Hugging Face revision to S3, once.

Standalone. If `s3://<bucket>/<prefix>/<revision>/MANIFEST.sha256` already exists the job ends
without downloading anything. Otherwise it downloads the revision (no token: the model is not
gated), uploads every file, and writes the manifest of sha256 sums last, so a manifest only
exists for a complete copy.

Usage: stage_weights.py <repo_id> <revision> <bucket> <prefix>
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

START = time.time()


def log(msg):
    print(f"stage: {msg} ({time.time() - START:.0f}s elapsed)", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(repo: str, revision: str, bucket: str, prefix: str, local: Path = Path("/opt/ml/processing/weights")) -> dict:
    import boto3
    s3 = boto3.client("s3")
    key = f"{prefix}/{revision}/MANIFEST.sha256"
    try:
        s3.head_object(Bucket=bucket, Key=key)
        log(f"s3://{bucket}/{key} exists: nothing to download")
        return {"downloaded": False, "manifest": key}
    except s3.exceptions.ClientError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.30"], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(local))
    files = sorted(p for p in local.rglob("*") if p.is_file() and ".cache" not in p.parts)
    log(f"downloaded {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB")
    lines = []
    for p in files:
        rel = p.relative_to(local).as_posix()
        lines.append(f"{sha256(p)}  {rel}")
        s3.upload_file(str(p), bucket, f"{prefix}/{revision}/{rel}")
    s3.put_object(Bucket=bucket, Key=key, Body=("\n".join(lines) + "\n").encode())
    info = {"downloaded": True, "files": len(files), "repo": repo, "revision": revision, "seconds": time.time() - START}
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{revision}/STAGED.json", Body=json.dumps(info).encode())
    log("uploaded; manifest written")
    return info


if __name__ == "__main__":
    main(*sys.argv[1:5])
