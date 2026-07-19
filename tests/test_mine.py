from __future__ import annotations

import json
from pathlib import Path

import torch

import gemma4_sae.mine as mine_module
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.mine import mine
from gemma4_sae.provenance import canonical_sha256, file_sha256, training_config_sha256
from gemma4_sae.sae import BatchTopKSAE
from gemma4_sae.storage import ActivationShardWriter


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


def test_mine_writes_provenance_and_control_contexts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    activation_dir = tmp_path / "activations"
    run_dir = tmp_path / "run"
    generator = torch.Generator().manual_seed(4)
    writer = ActivationShardWriter(
        activation_dir,
        d_model=4,
        tokens_per_shard=8,
        context_width=3,
        metadata={"model_id": "synthetic"},
    )
    writer.append(
        torch.randn(16, 4, generator=generator),
        torch.arange(16),
        torch.arange(48).reshape(16, 3),
    )
    manifest = writer.close()
    config = ProjectConfig(
        model=ModelConfig(backend="cpu", dtype="float32"),
        data=DataConfig(
            activation_dir=str(activation_dir),
            tokens_per_shard=8,
            context_radius=1,
        ),
        sae=SAEConfig(
            expansion_factor=2,
            target_l0=2,
            train_batch_size=4,
            validation_fraction=0.25,
            run_dir=str(run_dir),
        ),
    )
    sae = BatchTopKSAE(d_model=4, n_features=8, target_l0=2)
    sae.update_inference_threshold_(torch.tensor(0.0))
    checkpoint = {
        "step": 12,
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "sae_state_dict": sae.state_dict(),
        "activation_mean": torch.zeros(4),
        "activation_scale": torch.tensor(1.0),
        "target_l0": 2,
        "threshold_ema_decay": 0.999,
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    run_dir.mkdir()
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(mine_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        mine_module,
        "select_device",
        lambda _preference: torch.device("cpu"),
    )
    output = mine(
        config,
        str(checkpoint_path),
        [0, 1],
        n_features=2,
        top_contexts=2,
        random_contexts=2,
        max_batches=2,
    )

    report = json.loads(output.read_text())
    assert report["format_version"] == 2
    assert report["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert report["activation_manifest_sha256"] == canonical_sha256(manifest)
    assert report["mining_parameters"]["random_contexts"] == 2
    assert [feature["feature_id"] for feature in report["features"]] == [0, 1]
    assert all(
        len(feature["negative_contexts"]) <= 2
        for feature in report["features"]
    )
