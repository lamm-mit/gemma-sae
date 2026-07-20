from __future__ import annotations

import hashlib
import json
import mmap
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import Tensor

from .provenance import file_sha256

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
        hash_shards: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.d_model = d_model
        self.tokens_per_shard = tokens_per_shard
        self.context_width = context_width
        self.metadata = metadata
        self.hash_shards = hash_shards
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
        shard_record = {"stem": stem, "rows": self._position}
        if self.hash_shards:
            shard_record["sha256"] = {
                kind: file_sha256(self.root / f"{stem}.{kind}.npy")
                for kind in ("activations", "tokens", "contexts")
            }
        self.shards.append(shard_record)
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
            "shards_hashed": self.hash_shards,
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


def partition_shards(
    shards: list[dict],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Deterministically reserve whole shards, preventing adjacent-token leakage."""

    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5).")
    if validation_fraction == 0.0 or len(shards) < 2:
        return list(shards), []
    validation_count = max(1, round(len(shards) * validation_fraction))
    validation_count = min(validation_count, len(shards) - 1)

    ranked = sorted(
        shards,
        key=lambda shard: hashlib.sha256(
            f"{seed}:{shard['stem']}".encode()
        ).hexdigest(),
    )
    validation_stems = {shard["stem"] for shard in ranked[:validation_count]}
    training = [shard for shard in shards if shard["stem"] not in validation_stems]
    validation = [shard for shard in shards if shard["stem"] in validation_stems]
    return training, validation


def _release_mapped_file_cache(array: np.memmap, path: Path) -> None:
    """Release clean mmap pages after a shard pass, especially on unified-memory GPUs."""

    mapping = getattr(array, "_mmap", None)
    if (
        mapping is not None
        and hasattr(mapping, "madvise")
        and hasattr(mmap, "MADV_DONTNEED")
    ):
        try:
            mapping.madvise(mmap.MADV_DONTNEED)
        except (BufferError, OSError, ValueError):
            pass

    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        posix_fadvise(descriptor, 0, 0, dontneed)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def compute_training_statistics(
    root: str | Path,
    validation_fraction: float,
    seed: int,
    chunk_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute normalization from training shards without retaining their file cache."""

    root = Path(root)
    manifest = load_manifest(root)
    training_shards, _ = partition_shards(
        manifest["shards"],
        validation_fraction,
        seed,
    )
    d_model = int(manifest["d_model"])
    total = 0
    value_sum = np.zeros(d_model, dtype=np.float64)
    square_sum = np.zeros(d_model, dtype=np.float64)
    for shard in training_shards:
        rows = int(shard["rows"])
        path = root / f"{shard['stem']}.activations.npy"
        activations = np.load(path, mmap_mode="r")
        try:
            for start in range(0, rows, chunk_rows):
                values = np.asarray(
                    activations[start : min(start + chunk_rows, rows)]
                ).astype(np.float64)
                value_sum += values.sum(axis=0)
                square_sum += np.square(values).sum(axis=0)
                total += len(values)
        finally:
            _release_mapped_file_cache(activations, path)
            del activations
    if total == 0:
        raise RuntimeError("No training activation rows are available.")
    mean = value_sum / total
    variance = np.maximum(square_sum / total - np.square(mean), 0.0)
    global_rms = max(float(np.sqrt(variance.mean())), 1e-8)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.tensor(global_rms, dtype=torch.float32),
        total,
    )


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

    Validation reserves complete shards; training uses the remaining shards.
    """

    root = Path(root)
    manifest = load_manifest(root)
    epoch = 0
    while True:
        training_shards, validation_shards = partition_shards(
            manifest["shards"],
            validation_fraction,
            seed,
        )
        shards = validation_shards if validation else training_shards
        random.Random(seed + epoch).shuffle(shards)
        for shard in shards:
            stem = shard["stem"]
            rows = int(shard["rows"])
            rng = np.random.default_rng(seed + epoch * 1_000_003 + int(stem.split("-")[-1]))
            order = rng.permutation(np.arange(rows))
            activation_path = root / f"{stem}.activations.npy"
            token_path = root / f"{stem}.tokens.npy"
            context_path = root / f"{stem}.contexts.npy"
            activations = np.load(activation_path, mmap_mode="r")
            tokens = np.load(token_path, mmap_mode="r")
            contexts = np.load(context_path, mmap_mode="r")

            try:
                for offset in range(0, len(order), batch_size):
                    indices = order[offset : offset + batch_size]
                    if len(indices) < batch_size and not validation:
                        continue
                    yield ActivationBatch(
                        activations=torch.from_numpy(
                            np.asarray(activations[indices]).copy()
                        ),
                        token_ids=torch.from_numpy(np.asarray(tokens[indices]).copy()),
                        contexts=torch.from_numpy(np.asarray(contexts[indices]).copy()),
                    )
            finally:
                _release_mapped_file_cache(activations, activation_path)
                _release_mapped_file_cache(tokens, token_path)
                _release_mapped_file_cache(contexts, context_path)
                del activations, tokens, contexts
        if not repeat:
            return
        epoch += 1
