from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import Tensor

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ActivationBatch:
    activations: Tensor
    token_ids: Tensor
    contexts: Tensor


class ActivationShardWriter:
    """Write fixed-size NumPy shards and online activation statistics."""

    def __init__(
        self,
        root: str | Path,
        d_model: int,
        tokens_per_shard: int,
        context_width: int,
        metadata: dict,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.d_model = d_model
        self.tokens_per_shard = tokens_per_shard
        self.context_width = context_width
        self.metadata = metadata
        self.shards: list[dict] = []
        self.total_tokens = 0
        self.sum = np.zeros(d_model, dtype=np.float64)
        self.sum_squares = np.zeros(d_model, dtype=np.float64)
        self._shard_index = 0
        self._position = 0
        self._activations: np.memmap | None = None
        self._tokens: np.memmap | None = None
        self._contexts: np.memmap | None = None

    def _open_shard(self) -> None:
        stem = f"shard-{self._shard_index:06d}"
        self._activations = open_memmap(
            self.root / f"{stem}.activations.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self.tokens_per_shard, self.d_model),
        )
        self._tokens = open_memmap(
            self.root / f"{stem}.tokens.npy",
            mode="w+",
            dtype=np.int32,
            shape=(self.tokens_per_shard,),
        )
        self._contexts = open_memmap(
            self.root / f"{stem}.contexts.npy",
            mode="w+",
            dtype=np.int32,
            shape=(self.tokens_per_shard, self.context_width),
        )
        self._position = 0

    def append(self, activations: Tensor, token_ids: Tensor, contexts: Tensor) -> None:
        activations_np = activations.detach().float().cpu().numpy()
        token_ids_np = token_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        contexts_np = contexts.detach().cpu().numpy().astype(np.int32, copy=False)
        n_rows = len(activations_np)
        if token_ids_np.shape != (n_rows,):
            raise ValueError("token_ids must have shape [tokens].")
        if contexts_np.shape != (n_rows, self.context_width):
            raise ValueError(f"contexts must have shape [tokens, {self.context_width}].")
        if activations_np.shape != (n_rows, self.d_model):
            raise ValueError(f"activations must have shape [tokens, {self.d_model}].")

        self.sum += activations_np.astype(np.float64).sum(axis=0)
        self.sum_squares += np.square(activations_np.astype(np.float64)).sum(axis=0)
        self.total_tokens += n_rows

        offset = 0
        while offset < n_rows:
            if self._activations is None:
                self._open_shard()
            space = self.tokens_per_shard - self._position
            take = min(space, n_rows - offset)
            destination = slice(self._position, self._position + take)
            source = slice(offset, offset + take)
            self._activations[destination] = activations_np[source].astype(np.float16)
            self._tokens[destination] = token_ids_np[source]
            self._contexts[destination] = contexts_np[source]
            self._position += take
            offset += take
            if self._position == self.tokens_per_shard:
                self._close_shard()

    def _close_shard(self) -> None:
        if self._activations is None:
            return
        for array in (self._activations, self._tokens, self._contexts):
            array.flush()
        stem = f"shard-{self._shard_index:06d}"
        self.shards.append({"stem": stem, "rows": self._position})
        self._activations = None
        self._tokens = None
        self._contexts = None
        self._shard_index += 1
        self._position = 0

    def close(self) -> dict:
        self._close_shard()
        if self.total_tokens == 0:
            raise RuntimeError("No activations were written.")

        mean = self.sum / self.total_tokens
        mean_square = self.sum_squares / self.total_tokens
        variance = np.maximum(mean_square - np.square(mean), 0.0)
        global_rms = float(np.sqrt(variance.mean()))
        manifest = {
            "format_version": 1,
            "d_model": self.d_model,
            "context_width": self.context_width,
            "total_tokens": self.total_tokens,
            "activation_dtype": "float16",
            "token_dtype": "int32",
            "mean": mean.tolist(),
            "global_rms": global_rms,
            "shards": self.shards,
            "metadata": self.metadata,
        }
        temporary = self.root / f"{MANIFEST_NAME}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        temporary.replace(self.root / MANIFEST_NAME)
        return manifest

    def __enter__(self) -> ActivationShardWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


def load_manifest(root: str | Path) -> dict:
    path = Path(root) / MANIFEST_NAME
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported activation format: {manifest.get('format_version')}")
    return manifest


def iter_activation_batches(
    root: str | Path,
    batch_size: int,
    seed: int,
    *,
    validation: bool = False,
    validation_fraction: float = 0.0,
    repeat: bool = True,
) -> Iterator[ActivationBatch]:
    """Stream shuffled rows from memory-mapped shards.

    Validation uses the tail rows of every shard; training uses the remaining rows.
    """

    root = Path(root)
    manifest = load_manifest(root)
    epoch = 0
    while True:
        shards = list(manifest["shards"])
        random.Random(seed + epoch).shuffle(shards)
        for shard in shards:
            stem = shard["stem"]
            rows = int(shard["rows"])
            validation_rows = int(rows * validation_fraction)
            if validation:
                start, stop = rows - validation_rows, rows
            else:
                start, stop = 0, rows - validation_rows
            if stop <= start:
                continue

            rng = np.random.default_rng(seed + epoch * 1_000_003 + int(stem.split("-")[-1]))
            order = rng.permutation(np.arange(start, stop))
            activations = np.load(root / f"{stem}.activations.npy", mmap_mode="r")
            tokens = np.load(root / f"{stem}.tokens.npy", mmap_mode="r")
            contexts = np.load(root / f"{stem}.contexts.npy", mmap_mode="r")

            for offset in range(0, len(order), batch_size):
                indices = order[offset : offset + batch_size]
                if len(indices) < batch_size and not validation:
                    continue
                yield ActivationBatch(
                    activations=torch.from_numpy(np.asarray(activations[indices]).copy()),
                    token_ids=torch.from_numpy(np.asarray(tokens[indices]).copy()),
                    contexts=torch.from_numpy(np.asarray(contexts[indices]).copy()),
                )
        if not repeat:
            return
        epoch += 1
