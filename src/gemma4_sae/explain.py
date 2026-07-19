from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .devices import select_device
from .evaluate import build_sae_from_checkpoint
from .gemma import GemmaActivationExtractor, load_gemma
from .provenance import canonical_sha256
from .storage import load_manifest


def _decode_token(tokenizer, token_id: int) -> tuple[str, str]:
    token = tokenizer.convert_ids_to_tokens(token_id)
    try:
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    return str(token), str(text)


def summarize_prompt_features(
    tokenizer,
    input_ids: Tensor,
    features: Tensor,
    *,
    top_features_per_token: int,
    top_prompt_features: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence].")
    if features.ndim != 3 or features.shape[:2] != input_ids.shape:
        raise ValueError("features must have shape [1, sequence, n_features].")
    if top_features_per_token < 1 or top_prompt_features < 1:
        raise ValueError("Feature summary counts must be positive.")

    token_ids = input_ids[0].detach().cpu().long()
    prompt_features = features[0].detach().float().cpu()
    n_features = prompt_features.shape[-1]
    per_token_k = min(top_features_per_token, n_features)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    token_rows = []

    for position, token_id_tensor in enumerate(token_ids):
        token_id = int(token_id_tensor)
        values, indices = torch.topk(
            prompt_features[position],
            k=per_token_k,
            sorted=True,
        )
        active_features = [
            {"feature_id": int(feature_id), "activation": float(value)}
            for value, feature_id in zip(values, indices, strict=True)
            if float(value) > 0
        ]
        token, text = _decode_token(tokenizer, token_id)
        token_rows.append(
            {
                "position": position,
                "token_id": token_id,
                "token": token,
                "text": text,
                "special": token_id in special_ids,
                "active_feature_count": int((prompt_features[position] > 0).sum()),
                "top_features": active_features,
            }
        )

    active = prompt_features > 0
    counts = active.sum(dim=0)
    maxima = prompt_features.max(dim=0).values
    means = prompt_features.sum(dim=0) / counts.clamp_min(1)
    positive_features = (maxima > 0).nonzero(as_tuple=False).flatten()
    if positive_features.numel() == 0:
        return token_rows, []

    prompt_k = min(top_prompt_features, positive_features.numel())
    strongest = positive_features[
        torch.topk(maxima[positive_features], k=prompt_k, sorted=True).indices
    ]
    prompt_rows = [
        {
            "feature_id": int(feature_id),
            "max_activation": float(maxima[feature_id]),
            "mean_active_activation": float(means[feature_id]),
            "active_token_count": int(counts[feature_id]),
            "token_positions": active[:, feature_id].nonzero(as_tuple=False).flatten().tolist(),
        }
        for feature_id in strongest
    ]
    return token_rows, prompt_rows


def _context_examples(run_dir: Path, limit: int) -> dict[int, list[dict[str, Any]]]:
    report_path = run_dir / "feature_reports" / "features.json"
    if limit == 0 or not report_path.exists():
        return {}
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        int(feature["feature_id"]): [
            {
                "activation": context["activation"],
                "text": context["text"],
            }
            for context in feature.get("top_contexts", [])[:limit]
        ]
        for feature in report.get("features", [])
    }


def _attach_contexts(
    token_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    contexts: dict[int, list[dict[str, Any]]],
) -> None:
    for feature in prompt_rows:
        feature["known_contexts"] = contexts.get(feature["feature_id"], [])
    for token in token_rows:
        for feature in token["top_features"]:
            feature["known_contexts"] = contexts.get(feature["feature_id"], [])


@torch.inference_mode()
def explain_prompt(
    config: ProjectConfig,
    checkpoint_request: str,
    text: str,
    *,
    config_label: str,
    max_tokens: int,
    top_features_per_token: int,
    top_prompt_features: int,
    context_examples: int,
) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Prompt text must not be empty.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    if context_examples < 0:
        raise ValueError("context_examples must be non-negative.")

    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A trained checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)

    tokenizer, model = load_gemma(config.model)
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    with GemmaActivationExtractor(model, config.model.layer_index) as extractor:
        hidden = extractor.extract(input_ids, attention_mask)

    device = select_device(config.model.backend)
    sae = build_sae_from_checkpoint(checkpoint, device)
    mean = checkpoint["activation_mean"].float().to(device)
    scale = checkpoint["activation_scale"].float().to(device).clamp_min(1e-8)
    normalized = (hidden.float().to(device) - mean) / scale
    output = sae(normalized, use_threshold=True)
    token_rows, prompt_rows = summarize_prompt_features(
        tokenizer,
        input_ids,
        output.features,
        top_features_per_token=top_features_per_token,
        top_prompt_features=top_prompt_features,
    )

    run_dir = Path(config.sae.run_dir)
    contexts = _context_examples(run_dir, context_examples)
    _attach_contexts(token_rows, prompt_rows, contexts)
    feature_ids = [row["feature_id"] for row in prompt_rows]
    mine_command = (
        f"gemma4-sae mine --config {config_label} --checkpoint {checkpoint_request} "
        f"--features {' '.join(str(feature_id) for feature_id in feature_ids)} "
        "--top-contexts 40 --max-batches 4096"
        if feature_ids
        else None
    )

    token_l0 = [row["active_feature_count"] for row in token_rows]
    return {
        "format_version": 1,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "training_config_sha256": checkpoint["training_config_sha256"],
        "activation_manifest_sha256": canonical_sha256(manifest),
        "prompt": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(token_rows),
        "mean_inference_l0": sum(token_l0) / max(len(token_l0), 1),
        "inference_threshold": float(sae.inference_threshold),
        "tokens": token_rows,
        "prompt_features": prompt_rows,
        "context_examples_source": (
            str(run_dir / "feature_reports" / "features.json") if contexts else None
        ),
        "suggested_context_mining_command": mine_command,
    }


def add_explain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="latest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--top-features", type=int, default=5)
    parser.add_argument("--top-prompt-features", type=int, default=20)
    parser.add_argument(
        "--context-examples",
        type=int,
        default=3,
        help="Attach examples already present in feature_reports/features.json.",
    )
    parser.add_argument("--output", help="Optional JSON output path; prompt text is included.")


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    report = explain_prompt(
        load_config(args.config),
        args.checkpoint,
        text,
        config_label=args.config,
        max_tokens=args.max_tokens,
        top_features_per_token=args.top_features,
        top_prompt_features=args.top_prompt_features,
        context_examples=args.context_examples,
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(destination)
        print(f"Wrote {destination}.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a new prompt with a trained Gemma 4 SAE.",
    )
    add_explain_arguments(parser)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_from_args(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
