from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .gemma import read_hf_token
from .label import DEFAULT_REGISTRY_NAME, checkpoint_identity, load_label_registry
from .provenance import canonical_sha256, file_sha256, runtime_metadata
from .sae import BatchTopKSAE
from .storage import load_manifest

RELEASE_ARTIFACTS = (
    "evaluation.json",
    "fidelity.json",
    "run_metadata.json",
    "validation_metrics.json",
)
REQUIRED_RELEASE_FILES = (
    "activation_manifest.json",
    "release_metadata.json",
    "resolved_config.json",
    "sae_config.json",
    "sae_weights.safetensors",
)


def missing_release_evidence(run_dir: Path) -> list[str]:
    return [
        filename
        for filename in RELEASE_ARTIFACTS
        if not (run_dir / filename).exists()
    ]


def release_quality_failures(
    run_dir: Path,
    *,
    min_active_feature_fraction: float,
) -> list[str]:
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        return []
    evaluation_report = _read_json(evaluation_path)
    metrics = evaluation_report.get("metrics", evaluation_report)
    active_fraction = metrics.get("active_feature_fraction")
    if active_fraction is None:
        return ["evaluation.json lacks metrics.active_feature_fraction"]
    if float(active_fraction) < min_active_feature_fraction:
        return [
            "evaluation active_feature_fraction "
            f"{float(active_fraction):.4f} is below the configured publication minimum "
            f"{min_active_feature_fraction:.4f}"
        ]
    return []


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def verify_release_bundle(release_dir: str | Path) -> dict[str, str]:
    """Verify required files and every content checksum in an inference release."""

    release_dir = Path(release_dir)
    checksums_path = release_dir / "checksums.json"
    if not checksums_path.is_file():
        raise FileNotFoundError(f"Release checksum manifest is missing: {checksums_path}")
    checksums = _read_json(checksums_path)
    if not isinstance(checksums, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in checksums.items()
    ):
        raise ValueError("checksums.json must map release filenames to SHA-256 strings.")

    missing = [
        name
        for name in REQUIRED_RELEASE_FILES
        if not (release_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Release is missing required files: " + ", ".join(missing))

    required_without_manifest = set(REQUIRED_RELEASE_FILES)
    missing_checksums = sorted(required_without_manifest - set(checksums))
    if missing_checksums:
        raise ValueError(
            "checksums.json omits required files: " + ", ".join(missing_checksums)
        )

    for name, expected in checksums.items():
        if Path(name).name != name:
            raise ValueError(f"Release checksum entry must be a plain filename: {name!r}")
        path = release_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Checksummed release file is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"Release checksum mismatch for {name}: expected {expected}, got {observed}."
            )
    return checksums


def resolve_release_bundle(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local release directory or download a verified Hub model snapshot."""

    local_path = Path(source).expanduser()
    if local_path.is_dir():
        release_dir = local_path.resolve()
    else:
        release_dir = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="model",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
                token=read_hf_token(),
            )
        )
    verify_release_bundle(release_dir)
    return release_dir


def _model_card(
    config: ProjectConfig,
    checkpoint: dict,
    manifest: dict,
    metrics: dict[str, Any],
    repo_id: str,
) -> str:
    model_name = repo_id.split("/", maxsplit=1)[1]
    evaluation_report = metrics.get("evaluation.json", {})
    fidelity_report = metrics.get("fidelity.json", {})
    evaluation = evaluation_report.get("metrics", evaluation_report)
    fidelity = fidelity_report.get("metrics", fidelity_report)
    validation = metrics.get("validation_metrics.json", {})
    return f"""---
library_name: gemma4-sae
base_model: {config.model.model_id}
license: apache-2.0
tags:
- sparse-autoencoder
- mechanistic-interpretability
- gemma-4
- batchtopk
---

# {model_name}

Inference-only BatchTopK sparse-autoencoder weights for the output residual stream of
`{config.model.model_id}` decoder layer {config.model.layer_index}. The base-model
revision is `{config.model.revision}`.

## Intended use

This artifact supports interpretability research on the exact hook point and model
revision above. It is not a replacement language model. Feature labels are hypotheses,
not guaranteed uniquely true concepts.

## Architecture

- residual dimension: {manifest["d_model"]}
- dictionary width: {checkpoint["sae_state_dict"]["encoder.weight"].shape[0]}
- expansion factor: {config.sae.expansion_factor}
- training target L0: {config.sae.target_l0}
- auxiliary dead-latent top-k: {config.sae.auxiliary_top_k}
- auxiliary loss coefficient: {config.sae.auxiliary_loss_coefficient}
- inference threshold: {float(checkpoint["sae_state_dict"]["inference_threshold"]):.8g}
- activation normalization: subtract the released per-dimension mean, then divide by
  the released scalar global RMS
- decoder convention: unit-norm columns

## Data and provenance

- activation dataset: `{config.data.dataset_id}` / `{config.data.dataset_config}`
- dataset revision: `{config.data.revision}`
- activation split: `{config.data.split}`
- activation tokens: {manifest.get("total_tokens", "not recorded")}
- activation manifest SHA-256: `{canonical_sha256(manifest)}`
- training configuration SHA-256: `{checkpoint["training_config_sha256"]}`
- training step: {checkpoint["step"]}

## Reported metrics

```json
{json.dumps({"validation": validation, "evaluation": evaluation, "fidelity": fidelity}, indent=2)}
```

Consult `resolved_config.json`, `activation_manifest.json`, `run_metadata.json`, and
`checksums.json` before comparing this SAE with another release. Results are tied to the
exact hook convention, revisions, data mixture, normalization, width, L0, and seed.

## Loading

Explain a new prompt directly from this repository:

```bash
gemma4-sae explain \\
  --sae-repo {repo_id} \\
  --text "Paris is the capital of France." \\
  --output prompt-paris.json
```

The command downloads this verified inference release and the exact base-model revision,
then runs on CUDA, MPS, or CPU according to `--device` (default: `auto`).

For programmatic loading:

```python
from gemma4_sae.release import load_release_bundle, resolve_release_bundle

folder = resolve_release_bundle("{repo_id}")
sae, activation_mean, activation_scale, metadata = load_release_bundle(folder)
```

Normalize a hook activation as `(activation - activation_mean) / activation_scale`,
apply the SAE with `use_threshold=True`, then invert the normalization after decoding.

## Limitations

A sparse autoencoder is a learned decomposition and may split, merge, duplicate, or omit
concepts. Reconstruction metrics do not establish semantic interpretability. Validate
features on held-out data and with causal interventions. Mined text contexts are excluded
from the default bundle to reduce privacy and licensing risk. When `feature_labels.json`
is present, its descriptions are checkpoint-bound hypotheses with recorded automated
validation metrics; they are not ground-truth concepts.
"""


def load_release_bundle(
    release_dir: str | Path,
    device: str | torch.device = "cpu",
    *,
    verify: bool = True,
) -> tuple[BatchTopKSAE, torch.Tensor, torch.Tensor, dict[str, Any]]:
    release_dir = Path(release_dir)
    if verify:
        verify_release_bundle(release_dir)
    metadata = _read_json(release_dir / "sae_config.json")
    tensors = load_file(release_dir / "sae_weights.safetensors", device="cpu")
    state = {
        name.removeprefix("sae."): tensor
        for name, tensor in tensors.items()
        if name.startswith("sae.")
    }
    sae = BatchTopKSAE(
        d_model=int(metadata["d_model"]),
        n_features=int(metadata["n_features"]),
        target_l0=int(metadata["target_l0"]),
        threshold_ema_decay=float(metadata["threshold_ema_decay"]),
    )
    sae.load_state_dict(state)
    sae = sae.to(device).eval()
    return (
        sae,
        tensors["normalization.activation_mean"].to(device),
        tensors["normalization.activation_scale"].reshape(()).to(device),
        metadata,
    )


def build_release_bundle(
    config: ProjectConfig,
    checkpoint_request: str = "latest",
    *,
    repo_id: str | None = None,
) -> Path:
    run_dir = Path(config.sae.run_dir)
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A trained checkpoint is required for release packaging.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)

    step = int(checkpoint["step"])
    destination = repo_id or config.publication.hf_repo_id
    release_dir = run_dir / "release" / f"step-{step:08d}"
    release_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        f"sae.{name}": value.detach().cpu().contiguous()
        for name, value in checkpoint["sae_state_dict"].items()
        if torch.is_tensor(value)
    }
    weights["normalization.activation_mean"] = (
        checkpoint["activation_mean"].detach().cpu().contiguous()
    )
    weights["normalization.activation_scale"] = (
        checkpoint["activation_scale"].detach().cpu().reshape(1).contiguous()
    )
    weights_path = release_dir / "sae_weights.safetensors"
    save_file(
        weights,
        weights_path,
        metadata={
            "format": "gemma4-sae",
            "format_version": "1",
            "base_model": config.model.model_id,
            "base_model_revision": str(config.model.revision),
            "hook_point": f"text_decoder.layers.{config.model.layer_index}.output",
            "training_config_sha256": checkpoint["training_config_sha256"],
            "activation_manifest_sha256": canonical_sha256(manifest),
        },
    )

    sae_config = {
        "format_version": 1,
        "architecture": config.sae.architecture,
        "d_model": int(checkpoint["sae_state_dict"]["encoder.weight"].shape[1]),
        "n_features": int(checkpoint["sae_state_dict"]["encoder.weight"].shape[0]),
        "target_l0": int(checkpoint["target_l0"]),
        "auxiliary_top_k": config.sae.auxiliary_top_k,
        "auxiliary_loss_coefficient": config.sae.auxiliary_loss_coefficient,
        "threshold_ema_decay": float(checkpoint["threshold_ema_decay"]),
        "inference_threshold": float(
            checkpoint["sae_state_dict"]["inference_threshold"]
        ),
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "hook_point": f"text_decoder.layers.{config.model.layer_index}.output",
        "hook_output": "post-layer residual stream",
        "normalization": {
            "centering": "released per-dimension activation_mean",
            "scaling": "released scalar global RMS",
        },
        "checkpoint_step": step,
        "training_config_sha256": checkpoint["training_config_sha256"],
        "activation_manifest_sha256": canonical_sha256(manifest),
    }
    _write_json(release_dir / "sae_config.json", sae_config)
    _write_json(release_dir / "resolved_config.json", config.to_dict())
    _write_json(release_dir / "activation_manifest.json", manifest)
    _write_json(
        release_dir / "release_metadata.json",
        {
            "source_checkpoint": f"checkpoints/{checkpoint_path.name}",
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "hf_repo_id": destination,
            "release_runtime": runtime_metadata(),
            "contains_optimizer_state": False,
            "contains_mined_text_contexts": config.publication.include_feature_reports,
            "contains_feature_labels": (
                run_dir / DEFAULT_REGISTRY_NAME
            ).exists(),
        },
    )

    metrics: dict[str, Any] = {}
    for filename in RELEASE_ARTIFACTS:
        source = run_dir / filename
        destination_path = release_dir / filename
        if source.exists():
            shutil.copy2(source, destination_path)
            if filename.endswith(".json"):
                metrics[filename] = _read_json(source)
        elif destination_path.exists():
            destination_path.unlink()

    feature_report = run_dir / "feature_reports" / "features.json"
    release_feature_report = release_dir / "feature_reports.json"
    if config.publication.include_feature_reports:
        if not feature_report.exists():
            raise FileNotFoundError(
                "Publication requests feature reports, but feature_reports/features.json "
                "does not exist."
            )
        shutil.copy2(feature_report, release_feature_report)
    elif release_feature_report.exists():
        release_feature_report.unlink()

    label_registry = run_dir / DEFAULT_REGISTRY_NAME
    release_labels = release_dir / "feature_labels.json"
    if label_registry.exists():
        identity = checkpoint_identity(config, checkpoint, checkpoint_path, manifest)
        load_label_registry(label_registry, identity=identity)
        shutil.copy2(label_registry, release_labels)
    elif release_labels.exists():
        release_labels.unlink()

    (release_dir / "README.md").write_text(
        _model_card(config, checkpoint, manifest, metrics, destination),
        encoding="utf-8",
    )
    checksums = {
        path.name: file_sha256(path)
        for path in sorted(release_dir.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    _write_json(release_dir / "checksums.json", checksums)
    return release_dir


def publish_release(
    config: ProjectConfig,
    checkpoint_request: str = "latest",
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = repo_id or config.publication.hf_repo_id
    release_dir = build_release_bundle(
        config,
        checkpoint_request,
        repo_id=destination,
    )
    visibility = config.publication.private if private is None else private
    result: dict[str, Any] = {
        "release_dir": str(release_dir),
        "repo_id": destination,
        "private": visibility,
        "dry_run": dry_run,
    }
    if dry_run:
        result["missing_required_evidence"] = missing_release_evidence(
            Path(config.sae.run_dir)
        )
        result["quality_failures"] = release_quality_failures(
            Path(config.sae.run_dir),
            min_active_feature_fraction=config.publication.min_active_feature_fraction,
        )
        return result

    missing = missing_release_evidence(Path(config.sae.run_dir))
    if missing:
        raise RuntimeError(
            "Refusing to publish a run without required evidence: "
            + ", ".join(missing)
        )
    quality_failures = release_quality_failures(
        Path(config.sae.run_dir),
        min_active_feature_fraction=config.publication.min_active_feature_fraction,
    )
    if quality_failures:
        raise RuntimeError(
            "Refusing to publish a run that fails configured quality gates: "
            + "; ".join(quality_failures)
        )
    token = read_hf_token()
    if not token:
        raise RuntimeError("Set HF_TOKEN before publishing to Hugging Face.")
    api = HfApi(token=token)
    repo = api.create_repo(
        repo_id=destination,
        repo_type="model",
        private=visibility,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=destination,
        repo_type="model",
        folder_path=release_dir,
        commit_message=(
            f"Publish Gemma 4 layer-{config.model.layer_index} SAE "
            f"step {release_dir.name.removeprefix('step-')}"
        ),
    )
    result.update(
        {
            "repository_url": str(repo),
            "commit_url": str(commit.commit_url),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and publish an inference-only SAE release."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--repo-id", default=None)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true")
    visibility.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    private = True if args.private else False if args.public else None
    result = publish_release(
        load_config(args.config),
        args.checkpoint,
        repo_id=args.repo_id,
        private=private,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
