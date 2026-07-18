from pathlib import Path

from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.doctor import estimate_storage, filesystem_capacity


def test_storage_estimate_includes_contexts_and_checkpoints() -> None:
    config = ProjectConfig(
        model=ModelConfig(),
        data=DataConfig(
            max_activation_tokens=100,
            context_radius=2,
        ),
        sae=SAEConfig(
            expansion_factor=2,
            max_steps=10,
            checkpoint_every_steps=4,
        ),
    )
    estimate = estimate_storage(config, d_model=4)
    assert estimate["activation_array_bytes"] == 800
    assert estimate["token_array_bytes"] == 400
    assert estimate["context_array_bytes"] == 2_000
    assert estimate["activation_store_bytes"] == 3_200
    assert estimate["sae_parameters"] == 76
    assert estimate["estimated_checkpoint_count"] == 3
    assert estimate["estimated_checkpoint_bytes"] == 912


def test_filesystem_capacity_uses_nearest_existing_parent(tmp_path: Path) -> None:
    capacity = filesystem_capacity(tmp_path / "not-yet-created" / "run")
    assert capacity["checked_path"] == str(tmp_path.resolve())
    assert capacity["free_bytes"] > 0
