from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import torch


def _message_character_count(messages) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        len(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def tokenize_document(
    document: dict,
    tokenizer,
    *,
    column: str,
    input_format: str,
    min_chars: int,
) -> list[int] | None:
    value = document.get(column)
    if input_format == "text":
        if not isinstance(value, str) or len(value) < min_chars:
            return None
        return tokenizer(value, add_special_tokens=False)["input_ids"]
    if input_format == "messages":
        if _message_character_count(value) < min_chars:
            return None
        if not all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in value
        ):
            return None
        return tokenizer.apply_chat_template(
            value,
            tokenize=True,
            add_generation_prompt=False,
        )
    raise ValueError("input_format must be text or messages.")


def iter_token_blocks(
    documents: Iterable[dict],
    tokenizer,
    *,
    column: str,
    input_format: str,
    sequence_length: int,
    min_chars: int,
) -> Iterator[torch.Tensor]:
    """Pack text or templated conversations into fixed-length token blocks."""

    buffer: list[int] = []
    cursor = 0
    separator = tokenizer.eos_token_id
    if separator is None:
        raise ValueError("The tokenizer must define eos_token_id for document separation.")

    for document in documents:
        token_ids = tokenize_document(
            document,
            tokenizer,
            column=column,
            input_format=input_format,
            min_chars=min_chars,
        )
        if not token_ids:
            continue
        buffer.extend(token_ids)
        if buffer[-1] != separator:
            buffer.append(separator)

        while len(buffer) - cursor >= sequence_length:
            yield torch.tensor(buffer[cursor : cursor + sequence_length], dtype=torch.long)
            cursor += sequence_length
        if cursor >= 8 * sequence_length:
            buffer = buffer[cursor:]
            cursor = 0


def batched(items: Iterable[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield torch.stack(batch)
