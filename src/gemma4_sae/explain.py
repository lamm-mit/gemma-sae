from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .devices import select_device
from .evaluate import build_sae_from_checkpoint
from .gemma import GemmaActivationExtractor, load_gemma
from .label import (
    DEFAULT_REGISTRY_NAME,
    checkpoint_identity,
    label_lookup,
    load_label_registry,
    validate_feature_report,
)
from .provenance import canonical_sha256, training_config_sha256
from .release import load_release_bundle, resolve_release_bundle
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


def _context_examples(
    report_path: Path,
    limit: int,
    identity: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    if limit == 0 or not report_path.exists():
        return {}
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    validate_feature_report(report, identity)
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


def _load_interpretations(
    identity: dict[str, Any],
    requested: str | None,
    default_path: Path,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    if requested is None:
        return {}, None
    path = default_path if requested == "auto" else Path(requested)
    if not path.exists() and requested == "auto":
        return {}, None
    if not path.exists():
        raise FileNotFoundError(path)
    registry = load_label_registry(path, identity=identity)
    interpretations = {
        feature_id: {
            **record["interpretation"],
            "status": record["status"],
            "validation": record.get("validation"),
        }
        for feature_id, record in label_lookup(registry).items()
    }
    return interpretations, str(path)


def _attach_interpretations(
    token_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    interpretations: dict[int, dict[str, Any]],
) -> None:
    for feature in prompt_rows:
        feature["interpretation"] = interpretations.get(feature["feature_id"])
    for token in token_rows:
        for feature in token["top_features"]:
            feature["interpretation"] = interpretations.get(feature["feature_id"])


@torch.inference_mode()
def _explain_loaded_prompt(
    config: ProjectConfig,
    text: str,
    *,
    sae,
    activation_mean: Tensor,
    activation_scale: Tensor,
    identity: dict[str, Any],
    source: str,
    source_kind: str,
    context_report_path: Path,
    default_label_path: Path,
    mine_command_prefix: str | None,
    max_tokens: int,
    top_features_per_token: int,
    top_prompt_features: int,
    context_examples: int,
    label_registry: str | None = "auto",
) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Prompt text must not be empty.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    if context_examples < 0:
        raise ValueError("context_examples must be non-negative.")

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
    sae = sae.to(device).eval()
    mean = activation_mean.float().to(device)
    scale = activation_scale.float().to(device).clamp_min(1e-8)
    normalized = (hidden.float().to(device) - mean) / scale
    output = sae(normalized, use_threshold=True)
    token_rows, prompt_rows = summarize_prompt_features(
        tokenizer,
        input_ids,
        output.features,
        top_features_per_token=top_features_per_token,
        top_prompt_features=top_prompt_features,
    )

    contexts = _context_examples(context_report_path, context_examples, identity)
    _attach_contexts(token_rows, prompt_rows, contexts)
    interpretations, labels_source = _load_interpretations(
        identity,
        label_registry,
        default_label_path,
    )
    _attach_interpretations(token_rows, prompt_rows, interpretations)
    feature_ids = [row["feature_id"] for row in prompt_rows]
    mine_command = (
        f"{mine_command_prefix} "
        f"--features {' '.join(str(feature_id) for feature_id in feature_ids)} "
        "--top-contexts 40 --random-contexts 40 --max-batches 4096"
        if feature_ids and mine_command_prefix is not None
        else None
    )

    token_l0 = [row["active_feature_count"] for row in token_rows]
    labeled_prompt_features = sum(
        row["interpretation"] is not None for row in prompt_rows
    )
    return {
        "format_version": 1,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "sae_source_kind": source_kind,
        "checkpoint": source,
        "checkpoint_step": identity["checkpoint_step"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "training_config_sha256": identity["training_config_sha256"],
        "activation_manifest_sha256": identity["activation_manifest_sha256"],
        "prompt": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(token_rows),
        "mean_inference_l0": sum(token_l0) / max(len(token_l0), 1),
        "inference_threshold": float(sae.inference_threshold),
        "tokens": token_rows,
        "prompt_features": prompt_rows,
        "feature_label_registry": labels_source,
        "labeled_prompt_feature_fraction": (
            labeled_prompt_features / len(prompt_rows) if prompt_rows else 0.0
        ),
        "context_examples_source": (
            str(context_report_path) if contexts else None
        ),
        "suggested_context_mining_command": mine_command,
    }


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
    label_registry: str | None = "auto",
) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A trained checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)
    identity = checkpoint_identity(config, checkpoint, checkpoint_path, manifest)
    sae = build_sae_from_checkpoint(checkpoint, torch.device("cpu"))
    run_dir = Path(config.sae.run_dir)
    return _explain_loaded_prompt(
        config,
        text,
        sae=sae,
        activation_mean=checkpoint["activation_mean"],
        activation_scale=checkpoint["activation_scale"],
        identity=identity,
        source=str(checkpoint_path),
        source_kind="local_checkpoint",
        context_report_path=run_dir / "feature_reports" / "features.json",
        default_label_path=run_dir / DEFAULT_REGISTRY_NAME,
        mine_command_prefix=(
            f"gemma4-sae mine --config {config_label} --checkpoint {checkpoint_request}"
        ),
        max_tokens=max_tokens,
        top_features_per_token=top_features_per_token,
        top_prompt_features=top_prompt_features,
        context_examples=context_examples,
        label_registry=label_registry,
    )


def _required_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _release_checkpoint_identity(
    release_dir: Path,
    config: ProjectConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    manifest = _required_json(release_dir / "activation_manifest.json")
    release_metadata = _required_json(release_dir / "release_metadata.json")
    manifest_sha256 = canonical_sha256(manifest)
    expected_training_sha256 = training_config_sha256(config.to_dict())
    checks = {
        "model_id": (metadata.get("model_id"), config.model.model_id),
        "model_revision": (metadata.get("model_revision"), config.model.revision),
        "layer_index": (metadata.get("layer_index"), config.model.layer_index),
        "training_config_sha256": (
            metadata.get("training_config_sha256"),
            expected_training_sha256,
        ),
        "activation_manifest_sha256": (
            metadata.get("activation_manifest_sha256"),
            manifest_sha256,
        ),
    }
    mismatches = [
        field for field, (observed, expected) in checks.items() if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "Release metadata conflicts with resolved configuration or manifest: "
            + ", ".join(mismatches)
        )
    checkpoint_sha256 = release_metadata.get("source_checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError("release_metadata.json lacks source_checkpoint_sha256.")
    return {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "checkpoint_step": int(metadata["checkpoint_step"]),
        "checkpoint_sha256": checkpoint_sha256,
        "training_config_sha256": metadata["training_config_sha256"],
        "activation_manifest_sha256": manifest_sha256,
        "n_features": int(metadata["n_features"]),
    }


@torch.inference_mode()
def explain_release_prompt(
    release_source: str | Path,
    text: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    backend: str = "auto",
    dtype: str = "auto",
    load_in_4bit: bool = False,
    max_tokens: int,
    top_features_per_token: int,
    top_prompt_features: int,
    context_examples: int,
    label_registry: str | None = "auto",
) -> dict[str, Any]:
    release_dir = resolve_release_bundle(
        release_source,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    released_config = load_config(release_dir / "resolved_config.json")
    sae, mean, scale, metadata = load_release_bundle(
        release_dir,
        device="cpu",
        verify=False,
    )
    identity = _release_checkpoint_identity(release_dir, released_config, metadata)
    model_config = replace(
        released_config.model,
        backend=backend,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    config = replace(released_config, model=model_config)
    config.validate()
    return _explain_loaded_prompt(
        config,
        text,
        sae=sae,
        activation_mean=mean,
        activation_scale=scale,
        identity=identity,
        source=(
            f"hf://{release_source}@{revision or 'main'}"
            if not Path(release_source).expanduser().is_dir()
            else str(release_dir)
        ),
        source_kind="huggingface_release",
        context_report_path=release_dir / "feature_reports.json",
        default_label_path=release_dir / "feature_labels.json",
        mine_command_prefix=None,
        max_tokens=max_tokens,
        top_features_per_token=top_features_per_token,
        top_prompt_features=top_prompt_features,
        context_examples=context_examples,
        label_registry=label_registry,
    )


def add_explain_arguments(parser: argparse.ArgumentParser) -> None:
    sae_source = parser.add_mutually_exclusive_group(required=True)
    sae_source.add_argument("--config", help="Local training configuration.")
    sae_source.add_argument(
        "--sae-repo",
        help="Hugging Face SAE repository ID or local inference-release directory.",
    )
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--sae-revision", help="Optional Hugging Face SAE repository revision.")
    parser.add_argument("--cache-dir", help="Optional Hugging Face snapshot cache directory.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution backend for Hub releases (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Gemma dtype for Hub releases (default: auto).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load Gemma in 4-bit on CUDA; requires the quantization extra.",
    )
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
        help="Attach examples present in a compatible local or released feature report.",
    )
    parser.add_argument(
        "--labels",
        default="auto",
        help="Feature-label registry path, or auto for the local or released registry.",
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_const",
        const=None,
        help="Do not load reusable feature labels.",
    )
    parser.add_argument("--output", help="Optional JSON output path; prompt text is included.")


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    common = {
        "max_tokens": args.max_tokens,
        "top_features_per_token": args.top_features,
        "top_prompt_features": args.top_prompt_features,
        "context_examples": args.context_examples,
        "label_registry": args.labels,
    }
    if args.sae_repo:
        if args.checkpoint != "latest":
            raise ValueError("--checkpoint applies only with --config; use --sae-revision.")
        report = explain_release_prompt(
            args.sae_repo,
            text,
            revision=args.sae_revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            backend=args.device,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            **common,
        )
    else:
        if (
            args.sae_revision is not None
            or args.cache_dir is not None
            or args.local_files_only
            or args.device != "auto"
            or args.dtype != "auto"
            or args.load_in_4bit
        ):
            raise ValueError(
                "--sae-revision, --cache-dir, --local-files-only, --device, --dtype, "
                "and --load-in-4bit apply only with --sae-repo."
            )
        report = explain_prompt(
            load_config(args.config),
            args.checkpoint,
            text,
            config_label=args.config,
            **common,
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
