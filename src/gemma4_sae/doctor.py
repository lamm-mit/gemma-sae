from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectConfig, load_config
from .devices import resolve_model_dtype, select_device
from .gemma import read_hf_token
from .provenance import canonical_sha256, runtime_metadata
from .storage import MANIFEST_NAME


def diagnose(config: ProjectConfig) -> dict:
    device = select_device(config.model.backend)
    dtype = resolve_model_dtype(config.model.dtype, device)
    expected_d_model = 2560
    n_features = expected_d_model * config.sae.expansion_factor
    sae_parameters = (
        2 * expected_d_model * n_features
        + n_features
        + expected_d_model
    )
    activation_bytes = config.data.max_activation_tokens * expected_d_model * 2
    activation_dir = Path(config.data.activation_dir)
    return {
        "config_sha256": canonical_sha256(config.to_dict()),
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.data.dataset_id,
        "dataset_revision": config.data.revision,
        "backend": str(device),
        "model_dtype": str(dtype).replace("torch.", ""),
        "hf_token_available": bool(read_hf_token()),
        "estimated_activation_bytes": activation_bytes,
        "estimated_sae_parameters": sae_parameters,
        "activation_manifest_exists": (activation_dir / MANIFEST_NAME).exists(),
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
