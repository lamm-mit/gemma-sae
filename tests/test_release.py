import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import gemma4_sae.release as release_module
import gemma4_sae.train as train_module
from gemma4_sae.checkpoint import resolve_checkpoint
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.label import checkpoint_identity
from gemma4_sae.release import (
    build_release_bundle,
    load_release_bundle,
    publish_release,
    resolve_release_bundle,
    verify_release_bundle,
)
from gemma4_sae.storage import ActivationShardWriter, load_manifest


def _trained_tiny_config(tmp_path: Path, monkeypatch) -> ProjectConfig:
    activation_dir = tmp_path / "activations"
    run_dir = tmp_path / "run"
    writer = ActivationShardWriter(
        activation_dir,
        d_model=4,
        tokens_per_shard=32,
        context_width=3,
        metadata={"model_id": "synthetic"},
    )
    generator = torch.Generator().manual_seed(9)
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
            max_steps=2,
            warmup_steps=1,
            dead_after_steps=10,
            resample_every_steps=10,
            checkpoint_every_steps=2,
            log_every_steps=1,
            validation_fraction=0.25,
            run_dir=str(run_dir),
        ),
    )
    monkeypatch.setattr(
        train_module,
        "select_device",
        lambda _preference: torch.device("cpu"),
    )
    train_module.train(config)
    return config


def test_release_bundle_is_inference_only_and_hashed(tmp_path: Path, monkeypatch) -> None:
    config = _trained_tiny_config(tmp_path, monkeypatch)
    checkpoint_path = resolve_checkpoint(config.sae.run_dir, "latest")
    assert checkpoint_path is not None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    labels_path = Path(config.sae.run_dir) / "feature_labels" / "labels.json"
    labels_path.parent.mkdir()
    labels_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint": checkpoint_identity(
                    config,
                    checkpoint,
                    checkpoint_path,
                    manifest,
                ),
                "labels": [],
            }
        ),
        encoding="utf-8",
    )
    release_dir = build_release_bundle(config)

    assert (release_dir / "sae_weights.safetensors").exists()
    assert (release_dir / "README.md").exists()
    assert (release_dir / "feature_labels.json").exists()
    assert not (release_dir / "feature_reports.json").exists()
    with safe_open(release_dir / "sae_weights.safetensors", framework="pt") as handle:
        keys = set(handle.keys())
    assert "sae.encoder.weight" in keys
    assert "normalization.activation_mean" in keys
    assert not any("optimizer" in key for key in keys)
    checksums = json.loads((release_dir / "checksums.json").read_text())
    assert "sae_weights.safetensors" in checksums
    sae, mean, scale, metadata = load_release_bundle(release_dir)
    assert sae.n_features == 8
    assert mean.shape == (4,)
    assert scale.ndim == 0
    assert metadata["model_id"] == config.model.model_id
    assert resolve_release_bundle(release_dir) == release_dir.resolve()

    download_calls = {}

    def fake_snapshot_download(**kwargs):
        download_calls.update(kwargs)
        return str(release_dir)

    monkeypatch.setattr(release_module, "snapshot_download", fake_snapshot_download)
    assert (
        resolve_release_bundle("lamm-mit/test-sae", revision="release-commit")
        == release_dir
    )
    assert download_calls["repo_id"] == "lamm-mit/test-sae"
    assert download_calls["revision"] == "release-commit"

    (release_dir / "sae_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release_bundle(release_dir)


def test_publish_dry_run_never_calls_hugging_face(tmp_path: Path, monkeypatch) -> None:
    config = _trained_tiny_config(tmp_path, monkeypatch)
    result = publish_release(config, dry_run=True)
    assert result["repo_id"].startswith("lamm-mit/")
    assert result["dry_run"] is True
    assert "evaluation.json" in result["missing_required_evidence"]
    with pytest.raises(RuntimeError, match="required evidence"):
        publish_release(config)


def test_publish_uses_hugging_face_api_contract(tmp_path: Path, monkeypatch) -> None:
    config = _trained_tiny_config(tmp_path, monkeypatch)
    run_dir = Path(config.sae.run_dir)
    (run_dir / "evaluation.json").write_text(
        '{"metrics": {"active_feature_fraction": 1.0}}',
        encoding="utf-8",
    )
    (run_dir / "fidelity.json").write_text("{}", encoding="utf-8")
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs
            return "https://huggingface.co/lamm-mit/test-sae"

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs
            return SimpleNamespace(commit_url="https://huggingface.co/lamm-mit/test-sae/commit/1")

    monkeypatch.setattr(release_module, "HfApi", FakeApi)
    monkeypatch.setattr(release_module, "read_hf_token", lambda: "test-token")
    result = publish_release(
        config,
        repo_id="lamm-mit/test-sae",
        private=False,
    )
    assert result["repository_url"].endswith("lamm-mit/test-sae")
    assert result["commit_url"].endswith("/commit/1")
    assert calls["create_repo"]["private"] is False
    assert calls["upload_folder"]["repo_id"] == "lamm-mit/test-sae"


def test_publish_refuses_low_active_feature_fraction(tmp_path: Path, monkeypatch) -> None:
    config = _trained_tiny_config(tmp_path, monkeypatch)
    run_dir = Path(config.sae.run_dir)
    (run_dir / "evaluation.json").write_text(
        '{"metrics": {"active_feature_fraction": 0.25}}',
        encoding="utf-8",
    )
    (run_dir / "fidelity.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="quality gates"):
        publish_release(config)
