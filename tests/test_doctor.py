from pathlib import Path

from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.doctor import estimate_storage, filesystem_capacity, linux_memory_status


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


def test_linux_memory_status_parses_procfs_kibibytes(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       1024 kB\n"
        "MemFree:         128 kB\n"
        "MemAvailable:    900 kB\n"
        "Buffers:          16 kB\n"
        "Cached:          700 kB\n"
        "Ignored:          42 kB\n",
        encoding="utf-8",
    )

    assert linux_memory_status(path) == {
        "MemTotal": 1024 * 1024,
        "MemFree": 128 * 1024,
        "MemAvailable": 900 * 1024,
        "Buffers": 16 * 1024,
        "Cached": 700 * 1024,
    }
