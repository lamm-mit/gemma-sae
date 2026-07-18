from __future__ import annotations

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Resolve auto/cuda/mps/cpu without silently changing an explicit request."""

    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError("backend must be one of: auto, cuda, mps, cpu.")


def resolve_model_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Choose a model dtype that is usable on the selected backend."""

    requested = requested.lower()
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested not in choices:
        raise ValueError(f"dtype must be auto or one of {sorted(choices)}.")
    dtype = choices[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 model execution on CPU is unsupported; use auto or float32.")
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bfloat16 was requested, but this CUDA device does not support it.")
    return dtype
