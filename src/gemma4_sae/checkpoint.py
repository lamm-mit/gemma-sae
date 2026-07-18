from __future__ import annotations

import json
from pathlib import Path

import torch


def checkpoint_path(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step-{step:08d}.pt"


def save_checkpoint(run_dir: str | Path, step: int, payload: dict) -> Path:
    destination = checkpoint_path(run_dir, step)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)

    latest_path = destination.parent / "latest.json"
    latest_temporary = latest_path.with_suffix(".json.tmp")
    with latest_temporary.open("w", encoding="utf-8") as handle:
        json.dump({"step": step, "path": destination.name}, handle, indent=2)
    latest_temporary.replace(latest_path)
    return destination


def resolve_checkpoint(run_dir: str | Path, requested: str | None) -> Path | None:
    if requested is None:
        return None
    if requested != "latest":
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    latest_path = Path(run_dir) / "checkpoints" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No latest checkpoint at {latest_path}.")
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    path = latest_path.parent / latest["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    return path

