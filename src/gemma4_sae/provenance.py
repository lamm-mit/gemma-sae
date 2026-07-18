from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_config_sha256(config: dict) -> str:
    """Hash model/data/SAE settings while allowing independent evaluation protocols."""

    training_view = {
        key: config[key]
        for key in ("model", "data", "sae")
        if key in config
    }
    return canonical_sha256(training_view)


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    packages = ("accelerate", "datasets", "huggingface-hub", "numpy", "torch", "transformers")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def runtime_metadata(device: torch.device | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "software_versions": software_versions(),
    }
    if device is not None:
        result["device"] = str(device)
        if device.type == "cuda":
            result["accelerator_name"] = torch.cuda.get_device_name(device)
            result["cuda_version"] = torch.version.cuda
        elif device.type == "mps":
            result["accelerator_name"] = "Apple Metal Performance Shaders"
        else:
            result["accelerator_name"] = platform.processor() or "CPU"
    return result


def memory_metadata(device: torch.device) -> dict[str, int | str]:
    if device.type == "cuda":
        return {
            "unit": "bytes",
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
    if device.type == "mps":
        return {
            "unit": "bytes",
            "allocated": torch.mps.current_allocated_memory(),
            "driver_allocated": torch.mps.driver_allocated_memory(),
        }
    # ru_maxrss is bytes on macOS and KiB on Linux.
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {"unit": "bytes", "process_peak_rss": int(maximum_rss)}


def repository_commit(start: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
