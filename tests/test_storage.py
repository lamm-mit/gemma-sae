from pathlib import Path

import torch

from gemma4_sae.storage import (
    ActivationShardWriter,
    iter_activation_batches,
    load_manifest,
)


def test_shards_round_trip_and_preserve_partial_row_count(tmp_path: Path) -> None:
    writer = ActivationShardWriter(
        tmp_path,
        d_model=4,
        tokens_per_shard=5,
        context_width=3,
        metadata={"model_id": "test"},
    )
    activations = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    tokens = torch.arange(7)
    contexts = torch.arange(21).reshape(7, 3)
    writer.append(activations, tokens, contexts)
    manifest = writer.close()

    assert manifest["total_tokens"] == 7
    assert [shard["rows"] for shard in manifest["shards"]] == [5, 2]
    assert load_manifest(tmp_path)["d_model"] == 4

    batches = list(
        iter_activation_batches(
            tmp_path,
            batch_size=2,
            seed=0,
            validation=False,
            validation_fraction=0.0,
            repeat=False,
        )
    )
    # Training drops incomplete batches per shard: 4 rows from shard 0 and 2 from shard 1.
    assert sum(len(batch.activations) for batch in batches) == 6
    assert all(batch.contexts.shape[1] == 3 for batch in batches)


def test_validation_split_uses_held_out_tail(tmp_path: Path) -> None:
    writer = ActivationShardWriter(
        tmp_path,
        d_model=2,
        tokens_per_shard=10,
        context_width=1,
        metadata={},
    )
    writer.append(
        torch.arange(20, dtype=torch.float32).reshape(10, 2),
        torch.arange(10),
        torch.arange(10).reshape(10, 1),
    )
    writer.close()

    validation = list(
        iter_activation_batches(
            tmp_path,
            batch_size=2,
            seed=0,
            validation=True,
            validation_fraction=0.2,
            repeat=False,
        )
    )
    assert len(validation) == 1
    assert set(validation[0].token_ids.tolist()) == {8, 9}

