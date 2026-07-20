from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .devices import select_device
from .sae import BatchTopKSAE
from .storage import iter_activation_batches, load_manifest, partition_shards


def build_sae_from_checkpoint(checkpoint: dict, device: torch.device) -> BatchTopKSAE:
    model_state = checkpoint["sae_state_dict"]
    encoder_weight = model_state["encoder.weight"]
    d_model = encoder_weight.shape[1]
    n_features = encoder_weight.shape[0]
    target_l0 = int(checkpoint["target_l0"])
    sae = BatchTopKSAE(
        d_model=d_model,
        n_features=n_features,
        target_l0=target_l0,
        threshold_ema_decay=float(checkpoint["threshold_ema_decay"]),
    )
    sae.load_state_dict(model_state)
    return sae.to(device).eval()


@torch.inference_mode()
def evaluate_cached_activations(
    sae: BatchTopKSAE,
    config: ProjectConfig,
    *,
    activation_mean: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    max_batches: int = 64,
) -> dict[str, float | list[float]]:
    device = next(sae.parameters()).device
    if activation_mean is None or activation_scale is None:
        raise ValueError("Training-only activation normalization is required.")
    mean = activation_mean.float().cpu()
    scale = activation_scale.float().cpu().clamp_min(1e-8)
    iterator = iter_activation_batches(
        config.data.activation_dir,
        batch_size=config.sae.train_batch_size,
        seed=config.sae.seed,
        validation=True,
        validation_fraction=config.sae.validation_fraction,
        repeat=False,
    )
    manifest = load_manifest(config.data.activation_dir)
    _, validation_shards = partition_shards(
        manifest["shards"],
        config.sae.validation_fraction,
        config.sae.seed,
    )
    available_batches = sum(
        math.ceil(int(shard["rows"]) / config.sae.train_batch_size)
        for shard in validation_shards
    )
    evaluation_batches = min(max_batches, available_batches)

    squared_error = 0.0
    target_energy = 0.0
    cosine_sum = 0.0
    l0_sum = 0.0
    examples = 0
    feature_counts = torch.zeros(sae.n_features, dtype=torch.long, device=device)
    l0_values = []

    for batch in tqdm(
        islice(iterator, max_batches),
        total=evaluation_batches,
        desc="SAE evaluation",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        x = ((batch.activations.float() - mean) / scale).to(device)
        output = sae(x, use_threshold=True)
        squared_error += (output.reconstruction - x).square().sum().item()
        target_energy += x.square().sum().item()
        cosine_sum += F.cosine_similarity(output.reconstruction, x, dim=-1).sum().item()
        active = output.features > 0
        l0_batch = active.sum(dim=-1).float()
        l0_sum += l0_batch.sum().item()
        l0_values.append(l0_batch.cpu())
        feature_counts += active.sum(dim=0)
        examples += len(x)

    if examples == 0:
        raise RuntimeError(
            "No validation rows were available. Increase validation_fraction or collect more data."
        )
    dimensions = examples * sae.d_model
    l0_tensor = torch.cat(l0_values)
    feature_frequency = feature_counts.float().cpu() / examples
    return {
        "examples": float(examples),
        "normalized_mse": squared_error / dimensions,
        "fraction_variance_explained": 1.0 - squared_error / target_energy,
        "mean_cosine_similarity": cosine_sum / examples,
        "mean_l0": l0_sum / examples,
        "l0_quantiles_50_90_99": torch.quantile(
            l0_tensor,
            torch.tensor([0.5, 0.9, 0.99]),
        ).tolist(),
        "active_feature_fraction": (feature_counts > 0).float().mean().item(),
        "feature_frequency_quantiles_50_90_99": torch.quantile(
            feature_frequency,
            torch.tensor([0.5, 0.9, 0.99]),
        ).tolist(),
        "inference_threshold": sae.inference_threshold.item(),
    }


def evaluate(
    config: ProjectConfig,
    checkpoint_request: str,
    max_batches: int,
) -> dict[str, float | list[float]]:
    device = select_device(config.model.backend)
    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)
    sae = build_sae_from_checkpoint(checkpoint, device)
    metrics = evaluate_cached_activations(
        sae,
        config,
        activation_mean=checkpoint["activation_mean"],
        activation_scale=checkpoint["activation_scale"],
        max_batches=max_batches,
    )

    output_path = Path(config.sae.run_dir) / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"checkpoint": str(checkpoint_path), "metrics": metrics},
            handle,
            indent=2,
        )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Gemma 4 SAE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--max-batches", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(load_config(args.config), args.checkpoint, args.max_batches)


if __name__ == "__main__":
    main()
