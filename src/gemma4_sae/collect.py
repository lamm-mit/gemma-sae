from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .config import ProjectConfig, load_config
from .data import batched, iter_token_blocks
from .gemma import (
    GemmaActivationExtractor,
    contexts_for_tokens,
    load_gemma,
    valid_token_mask,
)
from .provenance import (
    canonical_sha256,
    memory_metadata,
    repository_commit,
    runtime_metadata,
)
from .storage import MANIFEST_NAME, ActivationShardWriter


def collect(config: ProjectConfig) -> dict:
    started = time.perf_counter()
    output_dir = Path(config.data.activation_dir)
    if (output_dir / MANIFEST_NAME).exists():
        raise FileExistsError(
            f"{output_dir / MANIFEST_NAME} already exists. "
            "Choose a new activation_dir to avoid mixing runs."
        )

    tokenizer, model = load_gemma(config.model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(
        config.data.dataset_id,
        name=config.data.dataset_config,
        split=config.data.split,
        streaming=True,
        revision=config.data.revision,
    )
    dataset = dataset.shuffle(
        seed=config.data.seed,
        buffer_size=config.data.shuffle_buffer,
    )
    blocks = iter_token_blocks(
        dataset,
        tokenizer,
        column=config.data.text_column,
        input_format=config.data.input_format,
        sequence_length=config.model.sequence_length,
        min_chars=config.data.min_chars,
    )

    text_config = model.config.get_text_config()
    context_width = 2 * config.data.context_radius + 1
    metadata = {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "requested_backend": config.model.backend,
        "resolved_backend": str(model.get_input_embeddings().weight.device),
        "requested_model_dtype": config.model.dtype,
        "resolved_model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "model_quantized_4bit": config.model.load_in_4bit,
        "sequence_length": config.model.sequence_length,
        "dataset_id": config.data.dataset_id,
        "dataset_config": config.data.dataset_config,
        "dataset_revision": config.data.revision,
        "dataset_split": config.data.split,
        "input_format": config.data.input_format,
        "seed": config.data.seed,
        "project_config_sha256": canonical_sha256(config.to_dict()),
        "repository_commit": repository_commit(Path(__file__).parents[2]),
        "runtime": runtime_metadata(model.get_input_embeddings().weight.device),
    }
    writer = ActivationShardWriter(
        output_dir,
        d_model=text_config.hidden_size,
        tokens_per_shard=config.data.tokens_per_shard,
        context_width=context_width,
        metadata=metadata,
        hash_shards=config.data.hash_shards,
    )

    progress = tqdm(total=config.data.max_activation_tokens, unit="tok", desc="Activations")
    special_ids = tokenizer.all_special_ids
    try:
        with GemmaActivationExtractor(model, config.model.layer_index) as extractor:
            print(
                f"Capturing {extractor.layer_path}[{config.model.layer_index}] "
                f"with hidden size {text_config.hidden_size}."
            )
            for input_ids in batched(blocks, config.model.inference_batch_size):
                attention_mask = torch.ones_like(input_ids)
                hidden = extractor.extract(input_ids, attention_mask).cpu()
                contexts = contexts_for_tokens(
                    input_ids,
                    radius=config.data.context_radius,
                    pad_token_id=pad_token_id,
                )
                mask = valid_token_mask(input_ids, attention_mask, special_ids)
                selected_activations = hidden[mask]
                selected_tokens = input_ids[mask]
                selected_contexts = contexts[mask]

                remaining = config.data.max_activation_tokens - writer.total_tokens
                take = min(remaining, len(selected_activations))
                writer.append(
                    selected_activations[:take],
                    selected_tokens[:take],
                    selected_contexts[:take],
                )
                progress.update(take)
                if writer.total_tokens >= config.data.max_activation_tokens:
                    break
    except Exception:
        progress.close()
        raise

    progress.close()
    writer.metadata["collection_elapsed_seconds"] = time.perf_counter() - started
    writer.metadata["activation_tokens_per_second"] = (
        writer.total_tokens / max(writer.metadata["collection_elapsed_seconds"], 1e-9)
    )
    writer.metadata["memory_at_collection_end"] = memory_metadata(
        model.get_input_embeddings().weight.device
    )
    manifest = writer.close()
    print(
        f"Wrote {manifest['total_tokens']:,} activations in "
        f"{len(manifest['shards'])} shards to {output_dir}."
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Gemma 4 residual-stream activations.")
    parser.add_argument("--config", required=True, help="Path to the YAML project configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(load_config(args.config))


if __name__ == "__main__":
    main()
