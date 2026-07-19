from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import (
    resolve_checkpoint,
    save_checkpoint,
    validate_checkpoint_provenance,
)
from .config import ProjectConfig, load_config
from .devices import select_device
from .evaluate import evaluate_cached_activations
from .provenance import (
    canonical_sha256,
    memory_metadata,
    repository_commit,
    runtime_metadata,
    training_config_sha256,
)
from .sae import BatchTopKSAE
from .storage import (
    compute_training_statistics,
    iter_activation_batches,
    load_manifest,
)


def cosine_multiplier(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def checkpoint_payload(
    config: ProjectConfig,
    step: int,
    sae: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_active_step: torch.Tensor,
    manifest: dict,
    device: torch.device,
    activation_mean: torch.Tensor,
    activation_scale: torch.Tensor,
) -> dict:
    payload = {
        "format_version": 1,
        "step": step,
        "config": config.to_dict(),
        "config_sha256": canonical_sha256(config.to_dict()),
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "activation_mean": activation_mean.cpu(),
        "activation_scale": activation_scale.cpu(),
        "repository_commit": repository_commit(Path(__file__).parents[2]),
        "runtime": runtime_metadata(device),
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "last_active_step": last_active_step.cpu(),
        "target_l0": sae.target_l0,
        "threshold_ema_decay": sae.threshold_ema_decay,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def restore_checkpoint(
    path: Path,
    sae: BatchTopKSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: ProjectConfig,
    manifest: dict,
) -> tuple[int, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    last_active = checkpoint["last_active_step"].to(device)
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return int(checkpoint["step"]), last_active


def train(config: ProjectConfig, resume: str | None = None) -> Path:
    device = select_device(config.model.backend)
    torch.manual_seed(config.sae.seed)
    np.random.seed(config.sae.seed)
    random.seed(config.sae.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    manifest = load_manifest(config.data.activation_dir)
    d_model = int(manifest["d_model"])
    n_features = d_model * config.sae.expansion_factor
    if config.sae.target_l0 > n_features:
        raise ValueError("target_l0 exceeds the SAE dictionary width.")

    mean, scale, normalization_rows = compute_training_statistics(
        config.data.activation_dir,
        config.sae.validation_fraction,
        config.sae.seed,
    )
    sae = BatchTopKSAE(
        d_model=d_model,
        n_features=n_features,
        target_l0=config.sae.target_l0,
        threshold_ema_decay=config.sae.threshold_ema_decay,
    ).to(device)
    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=config.sae.learning_rate,
        betas=(config.sae.beta1, config.sae.beta2),
        weight_decay=config.sae.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            config.sae.warmup_steps,
            config.sae.max_steps,
        ),
    )
    last_active_step = torch.zeros(n_features, dtype=torch.long, device=device)
    start_step = 0

    requested_checkpoint = resolve_checkpoint(config.sae.run_dir, resume)
    if requested_checkpoint is not None:
        start_step, last_active_step = restore_checkpoint(
            requested_checkpoint,
            sae,
            optimizer,
            scheduler,
            device,
            config,
            manifest,
        )
        print(f"Resumed {requested_checkpoint} at step {start_step}.")

    train_batches = iter_activation_batches(
        config.data.activation_dir,
        batch_size=config.sae.train_batch_size,
        seed=config.sae.seed,
        validation=False,
        validation_fraction=config.sae.validation_fraction,
        repeat=True,
    )
    run_dir = Path(config.sae.run_dir)
    metrics_path = run_dir / "train_metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    run_metadata = {
        "config_sha256": canonical_sha256(config.to_dict()),
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "activation_metadata": manifest.get("metadata", {}),
        "normalization_rows": normalization_rows,
        "activation_scale": scale.item(),
        "repository_commit": repository_commit(Path(__file__).parents[2]),
        "runtime": runtime_metadata(device),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    print(
        f"Training BatchTopK SAE on {device}: d_model={d_model}, features={n_features:,}, "
        f"target L0={config.sae.target_l0}, steps={config.sae.max_steps:,}, "
        f"auxiliary top-k={config.sae.auxiliary_top_k}, "
        f"auxiliary coefficient={config.sae.auxiliary_loss_coefficient:g}."
    )
    started = time.perf_counter()
    last_checkpoint = requested_checkpoint

    for step in range(start_step + 1, config.sae.max_steps + 1):
        batch = next(train_batches)
        x = ((batch.activations.float() - mean) / scale).to(device)
        sae.train()
        output = sae(x, use_threshold=False)
        active = output.selected_indices.unique()
        last_active_step[active] = step
        dead = (step - last_active_step) >= config.sae.dead_after_steps

        reconstruction_mse = F.mse_loss(output.reconstruction, x)
        auxiliary_mse = sae.auxiliary_dead_feature_loss(
            x,
            output.reconstruction,
            output.preactivations,
            dead,
            config.sae.auxiliary_top_k,
        )
        auxiliary_loss = config.sae.auxiliary_loss_coefficient * auxiliary_mse
        loss = reconstruction_mse + auxiliary_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_decoder_gradient_parallel_component_()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), config.sae.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder_()
        sae.update_inference_threshold_(output.batch_threshold)

        resampled = 0
        if (
            config.sae.resample_dead_features
            and step >= config.sae.dead_after_steps
            and step % config.sae.resample_every_steps == 0
        ):
            dead_indices = dead.nonzero(as_tuple=False).flatten()
            if len(dead_indices) > config.sae.max_resamples_per_event:
                permutation = torch.randperm(len(dead_indices), device=device)
                dead_indices = dead_indices[permutation[: config.sae.max_resamples_per_event]]
            residual = x - output.reconstruction.detach()
            resampled = sae.resample_dead_features_(dead_indices, residual, optimizer)
            last_active_step[dead_indices] = step

        if step == 1 or step % config.sae.log_every_steps == 0:
            with torch.no_grad():
                mean_l0 = (output.features > 0).sum(dim=-1).float().mean().item()
                fve = 1.0 - (output.reconstruction - x).square().sum() / x.square().sum()
                dead_fraction = (
                    (step - last_active_step) >= config.sae.dead_after_steps
                ).float().mean()
            record = {
                "step": step,
                "loss": loss.item(),
                "reconstruction_mse": reconstruction_mse.item(),
                "auxiliary_mse": auxiliary_mse.item(),
                "auxiliary_loss": auxiliary_loss.item(),
                "fraction_variance_explained": float(fve),
                "mean_l0": mean_l0,
                "dead_fraction": float(dead_fraction),
                "inference_threshold": sae.inference_threshold.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "resampled_features": resampled,
                "tokens_seen": step * config.sae.train_batch_size,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(metrics_path, record)
            print(
                f"step {step:7d} · mse {record['reconstruction_mse']:.5f} · "
                f"aux {record['auxiliary_loss']:.5f} · "
                f"FVE {record['fraction_variance_explained']:.3f} · "
                f"L0 {mean_l0:.1f} · dead {record['dead_fraction']:.2%}"
            )

        if step % config.sae.checkpoint_every_steps == 0 or step == config.sae.max_steps:
            payload = checkpoint_payload(
                config,
                step,
                sae,
                optimizer,
                scheduler,
                last_active_step,
                manifest,
                device,
                mean,
                scale,
            )
            last_checkpoint = save_checkpoint(run_dir, step, payload)
            print(f"Saved {last_checkpoint}.")

    if last_checkpoint is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    validation_metrics = evaluate_cached_activations(
        sae.eval(),
        config,
        activation_mean=mean,
        activation_scale=scale,
    )
    with (run_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, indent=2)
    run_metadata.update(
        {
            "training_elapsed_seconds": time.perf_counter() - started,
            "optimizer_examples_seen": config.sae.max_steps * config.sae.train_batch_size,
            "memory_at_training_end": memory_metadata(device),
        }
    )
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
    print(json.dumps(validation_metrics, indent=2))
    return last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BatchTopK SAE on cached activations.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint path, or from latest when passed without a value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
