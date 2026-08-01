"""Hugging Face Storage Bucket checkpoint transport.

Training keeps a small local cache for fast recovery, while a configured
``NLEARN_HF_BUCKET`` is the durable source of truth.  Storage Buckets are used
instead of model repositories because checkpoints are mutable runtime data,
not release artifacts that need Git history.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable


MANIFEST_NAME = "manifest.json"
LOCAL_MANIFEST_NAME = ".hf_manifest.json"
DEFAULT_HF_CONFIG = Path.home() / ".config" / "nlearn" / "hf.env"


def local_hf_config(path: str | Path | None = None) -> dict[str, str]:
    """Read the machine-local HF config without evaluating it as shell code."""
    config_path = Path(
        path or os.environ.get("NLEARN_HF_CONFIG", DEFAULT_HF_CONFIG)
    ).expanduser()
    values: dict[str, str] = {}
    try:
        lines = config_path.read_text().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key in {"HF_TOKEN", "HF_TOKEN_FILE", "NLEARN_HF_BUCKET", "NLEARN_HF_PREFIX"}:
            values[key] = value
    return values


def _setting(name: str) -> str | None:
    return os.environ.get(name) or local_hf_config().get(name)


def normalize_bucket_id(bucket_id: str) -> str:
    value = bucket_id.strip().rstrip("/")
    marker = "hf://buckets/"
    if value.startswith(marker):
        value = value[len(marker):]
    if value.count("/") != 1 or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(
            "HF bucket must be 'owner/name' or 'hf://buckets/owner/name', "
            f"got {bucket_id!r}"
        )
    return value


def bucket_id(explicit: str | None = None) -> str | None:
    value = explicit or _setting("NLEARN_HF_BUCKET")
    return normalize_bucket_id(value) if value else None


def _safe_run_name(run_name: str | None) -> str:
    value = run_name or "_default"
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", value):
        raise ValueError(
            "run name must contain only letters, numbers, '.', '_' and '-' "
            "when HF checkpointing is enabled"
        )
    return value


def _prefix(run_name: str | None) -> str:
    base = (_setting("NLEARN_HF_PREFIX") or "runs").strip("/")
    run = _safe_run_name(run_name)
    return f"{base}/{run}" if base else run


def _token() -> str | bool | None:
    token = _setting("HF_TOKEN")
    if token:
        return token
    token_file = _setting("HF_TOKEN_FILE")
    if token_file:
        try:
            return Path(token_file).expanduser().read_text().strip() or None
        except FileNotFoundError as exc:
            raise RuntimeError(f"HF_TOKEN_FILE does not exist: {token_file}") from exc
    return None


def _hf_functions():
    try:
        from huggingface_hub import batch_bucket_files, download_bucket_files
    except ImportError as exc:  # Give a useful error instead of an obscure import failure.
        raise RuntimeError(
            "HF checkpointing requires huggingface_hub>=1.8; install requirements.txt"
        ) from exc
    return batch_bucket_files, download_bucket_files


def _read_manifest(local_dir: Path, run_name: str | None) -> dict:
    path = local_dir / LOCAL_MANIFEST_NAME
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "format_version": 1,
        "run_name": run_name or "_default",
        "latest_step": None,
        "resume_step": None,
        "steps": [],
    }


def _write_local_manifest(local_dir: Path, manifest: dict) -> Path:
    path = local_dir / LOCAL_MANIFEST_NAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def _upload_manifest(bucket: str, run_name: str | None, path: Path) -> None:
    batch_bucket_files, _ = _hf_functions()
    batch_bucket_files(
        bucket,
        add=[(path, f"{_prefix(run_name)}/{MANIFEST_NAME}")],
        token=_token(),
    )


def upload_step_checkpoint(
    local_path: str | Path,
    step: int,
    run_name: str | None,
    retained_paths: Iterable[str | Path],
    removed_paths: Iterable[str | Path] = (),
) -> bool:
    """Synchronously upload a parameter checkpoint and publish it as latest.

    Returns False when no bucket is configured. Upload completion happens before
    the manifest is advanced, so inference never points at a partial upload.
    """
    bucket = bucket_id()
    if bucket is None:
        return False

    batch_bucket_files, _ = _hf_functions()
    local_path = Path(local_path)
    destination = f"{_prefix(run_name)}/steps/{local_path.name}"
    print(f"  Uploading checkpoint to hf://buckets/{bucket}/{destination} ...", flush=True)
    batch_bucket_files(bucket, add=[(local_path, destination)], token=_token())

    local_dir = local_path.parent
    manifest = _read_manifest(local_dir, run_name)
    manifest["latest_step"] = int(step)
    manifest["steps"] = [Path(path).name for path in retained_paths]
    manifest_path = _write_local_manifest(local_dir, manifest)
    _upload_manifest(bucket, run_name, manifest_path)

    # Retention cleanup is deliberately separate and best-effort. A failed delete
    # costs storage, while deleting before a confirmed upload risks data loss.
    remote_deletes = [f"{_prefix(run_name)}/steps/{Path(path).name}" for path in removed_paths]
    if remote_deletes:
        try:
            batch_bucket_files(bucket, delete=remote_deletes, token=_token())
        except Exception as exc:
            print(f"  Warning: remote checkpoint retention cleanup failed: {exc}", flush=True)
    print("  HF checkpoint upload complete.", flush=True)
    return True


def upload_resume_checkpoint(local_path: str | Path, step: int, run_name: str | None) -> bool:
    """Upload the mutable full training state used by ``--resume``."""
    bucket = bucket_id()
    if bucket is None:
        return False
    batch_bucket_files, _ = _hf_functions()
    local_path = Path(local_path)
    destination = f"{_prefix(run_name)}/resume.pkl"
    print(f"  Uploading resume state to hf://buckets/{bucket}/{destination} ...", flush=True)
    batch_bucket_files(bucket, add=[(local_path, destination)], token=_token())

    manifest = _read_manifest(local_path.parent, run_name)
    manifest["resume_step"] = int(step)
    manifest_path = _write_local_manifest(local_path.parent, manifest)
    _upload_manifest(bucket, run_name, manifest_path)
    print("  HF resume upload complete.", flush=True)
    return True


def _download(bucket: str, remote_path: str, local_path: Path) -> Path:
    _, download_bucket_files = _hf_functions()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        bucket,
        files=[(remote_path, local_path)],
        raise_on_missing_files=True,
        token=_token(),
    )
    if not local_path.exists():
        raise FileNotFoundError(f"HF checkpoint was not downloaded: {remote_path}")
    return local_path


def download_resume_checkpoint(local_path: str | Path, run_name: str | None) -> Path | None:
    """Restore ``resume.pkl`` from HF when no local copy exists."""
    bucket = bucket_id()
    if bucket is None:
        return None
    local_path = Path(local_path)
    tmp = local_path.with_suffix(local_path.suffix + ".download")
    try:
        _download(bucket, f"{_prefix(run_name)}/resume.pkl", tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        # ``--resume`` historically starts fresh when the run has no checkpoint.
        # Preserve that behavior while still surfacing auth/network failures.
        if exc.__class__.__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError"}:
            return None
        raise
    os.replace(tmp, local_path)
    print(f"  Restored resume state from HF: {local_path}", flush=True)
    return local_path


def download_inference_checkpoint(
    run_name: str,
    selector: str = "latest",
    bucket: str | None = None,
    destination_dir: str | Path | None = None,
) -> Path:
    """Download ``latest``, ``resume``, or a numbered step for inference."""
    resolved_bucket = bucket_id(bucket)
    if resolved_bucket is None:
        raise ValueError("Set NLEARN_HF_BUCKET or pass --hf-bucket owner/name")
    out_dir = Path(destination_dir or tempfile.mkdtemp(prefix="nlearn-checkpoint-"))
    prefix = _prefix(run_name)

    if selector == "latest":
        manifest_path = _download(
            resolved_bucket, f"{prefix}/{MANIFEST_NAME}", out_dir / MANIFEST_NAME
        )
        manifest = json.loads(manifest_path.read_text())
        steps = manifest.get("steps") or []
        if not steps:
            raise ValueError(f"HF checkpoint run {run_name!r} has no parameter checkpoints")
        filename = steps[-1]
        remote_path = f"{prefix}/steps/{filename}"
    elif selector == "resume":
        filename = "resume.pkl"
        remote_path = f"{prefix}/{filename}"
    else:
        value = selector.removeprefix("step_").removesuffix(".pkl")
        try:
            step = int(value)
        except ValueError as exc:
            raise ValueError("checkpoint must be 'latest', 'resume', or a step number") from exc
        filename = f"step_{step:06d}.pkl"
        remote_path = f"{prefix}/steps/{filename}"

    print(f"Downloading hf://buckets/{resolved_bucket}/{remote_path} ...", flush=True)
    return _download(resolved_bucket, remote_path, out_dir / filename)
