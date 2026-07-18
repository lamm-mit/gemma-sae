from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .config import ProjectConfig, load_config
from .gemma import (
    GemmaActivationExtractor,
    contexts_for_tokens,
    load_gemma,
    valid_token_mask,
)
from .storage import MANIFEST_NAME, ActivationShardWriter


def iter_token_blocks(
    documents: Iterable[dict],
    tokenizer,
    *,
    text_column: str,
    sequence_length: int,
    min_chars: int,
) -> Iterator[torch.Tensor]:
    buffer: list[int] = []
    separator = tokenizer.eos_token_id
    if separator is None:
        raise ValueError("The tokenizer must define eos_token_id for document separation.")

    for document in documents:
        text = document.get(text_column)
        if not isinstance(text, str) or len(text) < min_chars:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(token_ids)
        buffer.append(separator)
        while len(buffer) >= sequence_length:
            yield torch.tensor(buffer[:sequence_length], dtype=torch.long)
            del buffer[:sequence_length]


def batched(items: Iterable[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield torch.stack(batch)


def collect(config: ProjectConfig) -> dict:
    output_dir = Path(config.data.activation_dir)
    if (output_dir / MANIFEST_NAME).exists():
        raise FileExistsError(
            f"{output_dir / MANIFEST_NAME} already exists. "
            "Choose a new activation_dir to avoid mixing runs."
        )

    processor, model = load_gemma(config.model)
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(
        config.data.dataset_id,
        name=config.data.dataset_config,
        split=config.data.split,
        streaming=True,
    )
    dataset = dataset.shuffle(
        seed=config.data.seed,
        buffer_size=config.data.shuffle_buffer,
    )
    blocks = iter_token_blocks(
        dataset,
        tokenizer,
        text_column=config.data.text_column,
        sequence_length=config.model.sequence_length,
        min_chars=config.data.min_chars,
    )

    text_config = model.config.get_text_config()
    context_width = 2 * config.data.context_radius + 1
    metadata = {
        "model_id": config.model.model_id,
        "layer_index": config.model.layer_index,
        "model_dtype": config.model.dtype,
        "model_quantized_4bit": config.model.load_in_4bit,
        "sequence_length": config.model.sequence_length,
        "dataset_id": config.data.dataset_id,
        "dataset_config": config.data.dataset_config,
        "dataset_split": config.data.split,
        "seed": config.data.seed,
    }
    writer = ActivationShardWriter(
        output_dir,
        d_model=text_config.hidden_size,
        tokens_per_shard=config.data.tokens_per_shard,
        context_width=context_width,
        metadata=metadata,
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
