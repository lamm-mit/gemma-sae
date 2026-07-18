from __future__ import annotations

import argparse
import itertools
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .data import batched, iter_token_blocks
from .devices import select_device
from .evaluate import build_sae_from_checkpoint
from .gemma import find_text_decoder_layers, load_gemma
from .provenance import canonical_sha256, runtime_metadata
from .storage import load_manifest


class ResidualIntervention:
    """Replace a Gemma layer output with an SAE reconstruction or activation mean."""

    def __init__(
        self,
        model,
        layer_index: int,
        *,
        mode: str,
        sae,
        activation_mean: Tensor,
        activation_scale: Tensor,
    ) -> None:
        if mode not in {"sae", "mean"}:
            raise ValueError("mode must be sae or mean.")
        _, layers = find_text_decoder_layers(model)
        self.layer = layers[layer_index]
        self.mode = mode
        self.sae = sae
        self.sae_device = next(sae.parameters()).device
        self.mean = activation_mean.float().to(self.sae_device)
        self.scale = activation_scale.float().to(self.sae_device).clamp_min(1e-8)
        self.handle = None

    def __enter__(self) -> ResidualIntervention:
        def hook(_module, _inputs, output):
            is_sequence = isinstance(output, (tuple, list))
            hidden = output[0] if is_sequence else output
            original_device = hidden.device
            original_dtype = hidden.dtype
            if self.mode == "mean":
                replacement = self.mean.expand_as(hidden.to(self.sae_device))
            else:
                normalized = (hidden.float().to(self.sae_device) - self.mean) / self.scale
                sae_output = self.sae(normalized, use_threshold=True)
                replacement = sae_output.reconstruction * self.scale + self.mean
            replacement = replacement.to(device=original_device, dtype=original_dtype)
            if isinstance(output, tuple):
                return (replacement, *output[1:])
            if isinstance(output, list):
                return [replacement, *output[1:]]
            return replacement

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.inference_mode()
def mean_language_model_loss(
    model,
    blocks: list[Tensor],
    batch_size: int,
    intervention=None,
) -> tuple[float, int, list[float]]:
    if batch_size != 1:
        raise ValueError("Fidelity evaluation requires batch_size=1.")
    input_device = model.get_input_embeddings().weight.device
    loss_sum = 0.0
    predicted_tokens = 0
    sequence_losses = []
    context = intervention if intervention is not None else nullcontext()
    with context:
        for input_ids in batched(blocks, batch_size):
            input_ids = input_ids.to(input_device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
                return_dict=True,
            )
            count = int(attention_mask[:, 1:].sum())
            sequence_loss = float(outputs.loss)
            sequence_losses.append(sequence_loss)
            loss_sum += sequence_loss * count
            predicted_tokens += count
            del outputs
    return loss_sum / max(predicted_tokens, 1), predicted_tokens, sequence_losses


def bootstrap_intervals(
    baseline: list[float],
    sae: list[float],
    mean_ablation: list[float],
    *,
    seed: int,
    samples: int = 2_000,
) -> dict[str, list[float]]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    sae_array = np.asarray(sae, dtype=np.float64)
    mean_array = np.asarray(mean_ablation, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ce_increases = np.empty(samples)
    recovered = np.empty(samples)
    for sample in range(samples):
        indices = rng.integers(0, len(baseline_array), size=len(baseline_array))
        base_mean = baseline_array[indices].mean()
        sae_mean = sae_array[indices].mean()
        ablation_mean = mean_array[indices].mean()
        ce_increases[sample] = sae_mean - base_mean
        denominator = ablation_mean - base_mean
        recovered[sample] = (
            (ablation_mean - sae_mean) / denominator
            if abs(denominator) > 1e-12
            else np.nan
        )
    return {
        "sae_cross_entropy_increase_95ci": np.nanquantile(
            ce_increases,
            [0.025, 0.975],
        ).tolist(),
        "loss_recovered_95ci": np.nanquantile(
            recovered,
            [0.025, 0.975],
        ).tolist(),
    }


def fidelity(config: ProjectConfig, checkpoint_request: str) -> dict:
    device = select_device(config.model.backend)
    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)
    sae = build_sae_from_checkpoint(checkpoint, device)

    tokenizer, model = load_gemma(config.model)
    dataset = load_dataset(
        config.evaluation.dataset_id,
        name=config.evaluation.dataset_config,
        split=config.evaluation.split,
        streaming=True,
        revision=config.evaluation.revision,
    )
    block_iterator = iter_token_blocks(
        dataset,
        tokenizer,
        column=config.evaluation.text_column,
        input_format=config.evaluation.input_format,
        sequence_length=config.model.sequence_length,
        min_chars=config.evaluation.min_chars,
    )
    blocks = list(itertools.islice(block_iterator, config.evaluation.max_sequences))
    if not blocks:
        raise RuntimeError("The fidelity dataset produced no complete token blocks.")

    baseline_loss, predicted_tokens, baseline_sequence_losses = mean_language_model_loss(
        model,
        blocks,
        config.evaluation.batch_size,
    )
    sae_loss, _, sae_sequence_losses = mean_language_model_loss(
        model,
        blocks,
        config.evaluation.batch_size,
        intervention=ResidualIntervention(
            model,
            config.model.layer_index,
            mode="sae",
            sae=sae,
            activation_mean=checkpoint["activation_mean"],
            activation_scale=checkpoint["activation_scale"],
        ),
    )
    mean_ablation_loss, _, mean_sequence_losses = mean_language_model_loss(
        model,
        blocks,
        config.evaluation.batch_size,
        intervention=ResidualIntervention(
            model,
            config.model.layer_index,
            mode="mean",
            sae=sae,
            activation_mean=checkpoint["activation_mean"],
            activation_scale=checkpoint["activation_scale"],
        ),
    )
    denominator = mean_ablation_loss - baseline_loss
    loss_recovered = (
        (mean_ablation_loss - sae_loss) / denominator
        if abs(denominator) > 1e-12
        else float("nan")
    )
    metrics = {
        "baseline_cross_entropy": baseline_loss,
        "sae_cross_entropy": sae_loss,
        "mean_ablation_cross_entropy": mean_ablation_loss,
        "sae_cross_entropy_increase": sae_loss - baseline_loss,
        "loss_recovered": loss_recovered,
        "predicted_tokens": predicted_tokens,
        "sequences": len(blocks),
    }
    metrics.update(
        bootstrap_intervals(
            baseline_sequence_losses,
            sae_sequence_losses,
            mean_sequence_losses,
            seed=config.sae.seed,
        )
    )
    per_sequence = [
        {
            "sequence_index": index,
            "baseline_cross_entropy": baseline_value,
            "sae_cross_entropy": sae_value,
            "mean_ablation_cross_entropy": mean_value,
        }
        for index, (baseline_value, sae_value, mean_value) in enumerate(
            zip(
                baseline_sequence_losses,
                sae_sequence_losses,
                mean_sequence_losses,
                strict=True,
            )
        )
    ]
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_training_config_sha256": checkpoint["training_config_sha256"],
        "evaluation_config_sha256": canonical_sha256(config.to_dict()["evaluation"]),
        "evaluation_dataset": {
            "id": config.evaluation.dataset_id,
            "config": config.evaluation.dataset_config,
            "revision": config.evaluation.revision,
            "split": config.evaluation.split,
            "input_format": config.evaluation.input_format,
        },
        "runtime": runtime_metadata(device),
        "metrics": metrics,
        "per_sequence": per_sequence,
    }
    output_path = Path(config.sae.run_dir) / "fidelity.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure downstream SAE loss recovery.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fidelity(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
