from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from .checkpoint import resolve_checkpoint
from .config import ProjectConfig, load_config
from .evaluate import build_sae_from_checkpoint
from .gemma import read_hf_token
from .storage import iter_activation_batches, load_manifest


@torch.inference_mode()
def choose_candidate_features(
    sae,
    config: ProjectConfig,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    n_features: int,
    batches: int = 16,
) -> list[int]:
    device = next(sae.parameters()).device
    counts = torch.zeros(sae.n_features, device=device)
    strengths = torch.zeros(sae.n_features, device=device)
    examples = 0
    iterator = iter_activation_batches(
        config.data.activation_dir,
        batch_size=config.sae.train_batch_size,
        seed=config.sae.seed + 101,
        validation=False,
        validation_fraction=config.sae.validation_fraction,
        repeat=False,
    )
    for batch_index, batch in enumerate(iterator):
        if batch_index >= batches:
            break
        x = ((batch.activations.float() - mean) / scale).to(device)
        features, _, _ = sae.encode(x, use_threshold=True)
        counts += (features > 0).sum(dim=0)
        strengths += features.sum(dim=0)
        examples += len(x)

    frequency = counts / max(examples, 1)
    mean_strength = strengths / counts.clamp_min(1)
    valid = (frequency >= 1e-4) & (frequency <= 0.10)
    score = mean_strength * torch.sqrt(frequency.clamp_min(0))
    score[~valid] = -torch.inf
    available = int(valid.sum())
    if available == 0:
        raise RuntimeError("No candidate features fired in the selection sample.")
    return torch.topk(score, k=min(n_features, available)).indices.cpu().tolist()


@torch.inference_mode()
def mine(
    config: ProjectConfig,
    checkpoint_request: str,
    feature_ids: list[int] | None,
    *,
    n_features: int,
    top_contexts: int,
    max_batches: int,
) -> Path:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae = build_sae_from_checkpoint(checkpoint, device)
    manifest = load_manifest(config.data.activation_dir)
    mean = torch.tensor(manifest["mean"], dtype=torch.float32)
    scale = torch.tensor(manifest["global_rms"], dtype=torch.float32).clamp_min(1e-8)

    if feature_ids is None:
        feature_ids = choose_candidate_features(
            sae,
            config,
            mean,
            scale,
            n_features=n_features,
        )
    invalid = [feature_id for feature_id in feature_ids if not 0 <= feature_id < sae.n_features]
    if invalid:
        raise ValueError(f"Feature IDs out of range: {invalid}")

    heaps: dict[int, list[tuple[float, list[int]]]] = {
        feature_id: [] for feature_id in feature_ids
    }
    frequencies = {feature_id: 0 for feature_id in feature_ids}
    examples = 0
    iterator = iter_activation_batches(
        config.data.activation_dir,
        batch_size=config.sae.train_batch_size,
        seed=config.sae.seed + 211,
        validation=False,
        validation_fraction=config.sae.validation_fraction,
        repeat=False,
    )

    for batch_index, batch in enumerate(iterator):
        if batch_index >= max_batches:
            break
        x = ((batch.activations.float() - mean) / scale).to(device)
        features, _, _ = sae.encode(x, use_threshold=True)
        selected = features[:, feature_ids].float().cpu()
        examples += len(x)

        for column, feature_id in enumerate(feature_ids):
            values = selected[:, column]
            frequencies[feature_id] += int((values > 0).sum())
            nonzero = (values > 0).nonzero(as_tuple=False).flatten()
            for row in nonzero.tolist():
                item = (float(values[row]), batch.contexts[row].tolist())
                heap = heaps[feature_id]
                if len(heap) < top_contexts:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    processor = AutoProcessor.from_pretrained(
        config.model.model_id,
        token=read_hf_token(),
    )
    tokenizer = processor.tokenizer
    report = {
        "checkpoint": str(checkpoint_path),
        "examples_scanned": examples,
        "features": [],
    }
    for feature_id in feature_ids:
        contexts = [
            {
                "activation": activation,
                "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "token_ids": token_ids,
            }
            for activation, token_ids in sorted(heaps[feature_id], reverse=True)
        ]
        report["features"].append(
            {
                "feature_id": feature_id,
                "activation_frequency": frequencies[feature_id] / max(examples, 1),
                "top_contexts": contexts,
            }
        )

    output_dir = Path(config.sae.run_dir) / "feature_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path} with {len(feature_ids)} feature reports.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine top token contexts for SAE features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--features", type=int, nargs="*", default=None)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--top-contexts", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mine(
        load_config(args.config),
        args.checkpoint,
        args.features,
        n_features=args.n_features,
        top_contexts=args.top_contexts,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
