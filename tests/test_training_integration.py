from pathlib import Path

import torch

import gemma4_sae.train as train_module
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.storage import ActivationShardWriter


def test_tiny_training_run_produces_checkpoint_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    activation_dir = tmp_path / "activations"
    run_dir = tmp_path / "run"
    writer = ActivationShardWriter(
        activation_dir,
        d_model=4,
        tokens_per_shard=32,
        context_width=3,
        metadata={"model_id": "synthetic"},
    )
    generator = torch.Generator().manual_seed(3)
    writer.append(
        torch.randn(64, 4, generator=generator),
        torch.arange(64),
        torch.zeros(64, 3, dtype=torch.long),
    )
    writer.close()

    config = ProjectConfig(
        model=ModelConfig(),
        data=DataConfig(
            activation_dir=str(activation_dir),
            max_activation_tokens=64,
            tokens_per_shard=32,
            context_radius=1,
        ),
        sae=SAEConfig(
            expansion_factor=2,
            target_l0=2,
            train_batch_size=8,
            learning_rate=1e-3,
            max_steps=3,
            warmup_steps=1,
            dead_after_steps=10,
            resample_every_steps=10,
            checkpoint_every_steps=3,
            log_every_steps=1,
            validation_fraction=0.25,
            run_dir=str(run_dir),
        ),
    )
    monkeypatch.setattr(train_module, "select_device", lambda: torch.device("cpu"))
    checkpoint = train_module.train(config)

    assert checkpoint.exists()
    assert (run_dir / "train_metrics.jsonl").exists()
    assert (run_dir / "validation_metrics.json").exists()
