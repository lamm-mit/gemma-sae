from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
from pathlib import Path

import torch

from .config import ProjectConfig, load_config
from .devices import resolve_model_dtype, select_device
from .gemma import read_hf_token
from .provenance import canonical_sha256, runtime_metadata
from .storage import MANIFEST_NAME


def estimate_storage(config: ProjectConfig, d_model: int = 2560) -> dict[str, int]:
    tokens = config.data.max_activation_tokens
    context_width = 2 * config.data.context_radius + 1
    n_features = d_model * config.sae.expansion_factor
    sae_parameters = 2 * d_model * n_features + n_features + d_model

    activation_bytes = tokens * d_model * 2
    token_bytes = tokens * 4
    context_bytes = tokens * context_width * 4
    activation_store_bytes = activation_bytes + token_bytes + context_bytes

    # Checkpoints contain FP32 SAE parameters and two AdamW moment tensors. Small
    # scheduler, RNG, and feature-activity tensors are omitted from this estimate.
    checkpoint_bytes = sae_parameters * 4 * 3
    checkpoint_count = math.ceil(
        config.sae.max_steps / config.sae.checkpoint_every_steps
    )
    checkpoint_store_bytes = checkpoint_count * checkpoint_bytes
    return {
        "activation_array_bytes": activation_bytes,
        "token_array_bytes": token_bytes,
        "context_array_bytes": context_bytes,
        "activation_store_bytes": activation_store_bytes,
        "sae_parameters": sae_parameters,
        "sae_parameter_bytes_fp32": sae_parameters * 4,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_checkpoint_count": checkpoint_count,
        "estimated_checkpoint_store_bytes": checkpoint_store_bytes,
        "estimated_persistent_bytes": activation_store_bytes + checkpoint_store_bytes,
    }


def filesystem_capacity(path: str | Path) -> dict[str, int | str]:
    requested = Path(path).expanduser().resolve()
    existing = requested
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {
        "requested_path": str(requested),
        "checked_path": str(existing),
        "device": int(existing.stat().st_dev),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def diagnose(config: ProjectConfig) -> dict:
    device = select_device(config.model.backend)
    dtype = resolve_model_dtype(config.model.dtype, device)
    expected_d_model = 2560
    storage = estimate_storage(config, expected_d_model)
    activation_dir = Path(config.data.activation_dir)
    activation_filesystem = filesystem_capacity(activation_dir)
    run_filesystem = filesystem_capacity(config.sae.run_dir)
    warnings = []

    safety_factor = 1.25
    model_cache_reserve = 32 * 1024**3
    storage["recommended_shared_filesystem_free_bytes"] = (
        int(storage["estimated_persistent_bytes"] * safety_factor)
        + model_cache_reserve
    )
    if activation_filesystem["device"] == run_filesystem["device"]:
        required = storage["recommended_shared_filesystem_free_bytes"]
        if activation_filesystem["free_bytes"] < required:
            warnings.append(
                "The activation and run directories share a filesystem with less than "
                "125% of the estimated persistent-storage requirement plus a 32 GiB "
                "model-cache reserve."
            )
    else:
        activation_required = int(storage["activation_store_bytes"] * safety_factor)
        checkpoint_required = int(storage["estimated_checkpoint_store_bytes"] * safety_factor)
        if activation_filesystem["free_bytes"] < activation_required:
            warnings.append(
                "The activation filesystem has less than 125% of the estimated "
                "activation-store requirement."
            )
        if run_filesystem["free_bytes"] < checkpoint_required:
            warnings.append(
                "The run filesystem has less than 125% of the estimated checkpoint "
                "requirement."
            )

    accelerator = {}
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        accelerator = {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
            "is_dgx_spark_gb10": capability == (12, 1)
            or "GB10" in torch.cuda.get_device_name(device).upper(),
        }
        if config.model.dtype == "bfloat16" and not accelerator["bfloat16_supported"]:
            warnings.append("The configuration requires BF16, but CUDA reports no BF16 support.")

    return {
        "config_sha256": canonical_sha256(config.to_dict()),
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.data.dataset_id,
        "dataset_revision": config.data.revision,
        "backend": str(device),
        "model_dtype": str(dtype).replace("torch.", ""),
        "hf_token_available": bool(read_hf_token()),
        "host_architecture": platform.machine(),
        "accelerator": accelerator,
        "estimated_activation_bytes": storage["activation_array_bytes"],
        "estimated_sae_parameters": storage["sae_parameters"],
        "storage": storage,
        "activation_filesystem": activation_filesystem,
        "run_filesystem": run_filesystem,
        "activation_manifest_exists": (activation_dir / MANIFEST_NAME).exists(),
        "warnings": warnings,
        "runtime": runtime_metadata(device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a config and estimate resources.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(diagnose(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
