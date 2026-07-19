from __future__ import annotations

import json
from pathlib import Path

import torch

import gemma4_sae.explain as explain_module
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.label import checkpoint_identity
from gemma4_sae.provenance import canonical_sha256, training_config_sha256
from gemma4_sae.sae import BatchTopKSAE


class FakeTokenizer:
    all_special_ids = [0]

    def __call__(self, _text, **_kwargs):
        return {
            "input_ids": torch.tensor([[0, 11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, token_id):
        return {0: "<bos>", 11: "Paris", 12: "ĠFrance"}[token_id]

    def decode(self, token_ids, **_kwargs):
        return {0: "<bos>", 11: "Paris", 12: " France"}[token_ids[0]]


class FakeExtractor:
    def __init__(self, _model, _layer_index):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        pass

    def extract(self, _input_ids, _attention_mask):
        return torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.5, -0.5, 0.25],
                    [0.5, 1.0, 0.25, -0.5],
                ]
            ]
        )


def test_feature_summary_reports_tokens_and_prompt_features() -> None:
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([[0, 11, 12]])
    features = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 2.0, 4.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    tokens, prompt_features = explain_module.summarize_prompt_features(
        tokenizer,
        input_ids,
        features,
        top_features_per_token=2,
        top_prompt_features=2,
    )

    assert tokens[0]["special"] is True
    assert tokens[1]["text"] == "Paris"
    assert tokens[1]["top_features"][0] == {"feature_id": 1, "activation": 3.0}
    assert prompt_features[0]["feature_id"] == 2
    assert prompt_features[0]["token_positions"] == [2]
    assert prompt_features[1]["feature_id"] == 1
    assert prompt_features[1]["token_positions"] == [1, 2]


def test_explain_prompt_runs_with_pinned_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    activation_dir = tmp_path / "activations"
    run_dir = tmp_path / "run"
    activation_dir.mkdir()
    run_dir.mkdir()
    manifest = {
        "format_version": 1,
        "d_model": 4,
        "total_tokens": 3,
        "shards": [],
        "metadata": {"model_id": "synthetic"},
    }
    (activation_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = ProjectConfig(
        model=ModelConfig(backend="cpu", dtype="float32"),
        data=DataConfig(activation_dir=str(activation_dir)),
        sae=SAEConfig(
            expansion_factor=2,
            target_l0=2,
            run_dir=str(run_dir),
        ),
    )
    sae = BatchTopKSAE(d_model=4, n_features=8, target_l0=2)
    sae.update_inference_threshold_(torch.tensor(0.0))
    checkpoint = {
        "step": 25,
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "sae_state_dict": sae.state_dict(),
        "activation_mean": torch.zeros(4),
        "activation_scale": torch.tensor(1.0),
        "target_l0": 2,
        "threshold_ema_decay": 0.999,
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    labels_dir = run_dir / "feature_labels"
    labels_dir.mkdir()
    identity = checkpoint_identity(config, checkpoint, checkpoint_path, manifest)
    (labels_dir / "labels.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint": identity,
                "labels": [
                    {
                        "feature_id": feature_id,
                        "status": "candidate",
                        "interpretation": {
                            "label": f"synthetic feature {feature_id}",
                            "description": "Synthetic test label.",
                            "activation_rule": "Used only in tests.",
                            "confidence": "low",
                            "polysemantic": False,
                            "facets": [],
                            "caveats": [],
                        },
                        "validation": None,
                    }
                    for feature_id in range(8)
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        explain_module,
        "load_gemma",
        lambda _config: (FakeTokenizer(), object()),
    )
    monkeypatch.setattr(explain_module, "GemmaActivationExtractor", FakeExtractor)
    monkeypatch.setattr(
        explain_module,
        "select_device",
        lambda _backend: torch.device("cpu"),
    )

    report = explain_module.explain_prompt(
        config,
        str(checkpoint_path),
        "Paris France",
        config_label="config.yaml",
        max_tokens=16,
        top_features_per_token=3,
        top_prompt_features=4,
        context_examples=0,
    )

    assert report["checkpoint_step"] == 25
    assert report["prompt"] == "Paris France"
    assert report["token_count"] == 3
    assert report["tokens"][1]["text"] == "Paris"
    assert report["prompt_features"]
    assert report["feature_label_registry"].endswith("feature_labels/labels.json")
    assert report["prompt_features"][0]["interpretation"]["label"].startswith(
        "synthetic feature"
    )
    assert report["labeled_prompt_feature_fraction"] == 1.0
    assert "--features" in report["suggested_context_mining_command"]
