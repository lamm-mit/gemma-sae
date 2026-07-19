from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

import gemma4_sae.label as label_module
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.label import (
    AnthropicMessagesModel,
    ModelResult,
    OpenAIResponsesModel,
    ProviderSpec,
    RetryableProviderResponseError,
    add_label_arguments,
    checkpoint_identity,
    label_features,
    load_label_registry,
    provider_spec_from_args,
)
from gemma4_sae.provenance import canonical_sha256, file_sha256, training_config_sha256


class FakeJSONModel(label_module.JSONModel):
    def __init__(self):
        self.calls = []

    def generate(self, *, system, prompt, schema, schema_name):
        self.calls.append((system, prompt, schema, schema_name))
        if schema_name == "sae_feature_interpretation":
            return ModelResult(
                {
                    "label": "commercial lending",
                    "description": "Financial institutions granting loans or credit.",
                    "activation_rule": "Activates on lending and credit approval language.",
                    "confidence": "high",
                    "polysemantic": False,
                    "facets": ["banks", "loans"],
                    "caveats": ["May include adjacent financial language."],
                },
                {"response_id": "label-1", "resolved_model": "fake", "usage": {}},
            )
        return ModelResult(
            {
                "predictions": [
                    {"example_id": "e000", "predicted_activation": 4},
                    {"example_id": "e001", "predicted_activation": 0},
                ]
            },
            {"response_id": "score-1", "resolved_model": "fake", "usage": {}},
        )


def _label_fixture(tmp_path: Path):
    activation_dir = tmp_path / "activations"
    run_dir = tmp_path / "run"
    activation_dir.mkdir()
    run_dir.mkdir()
    manifest = {
        "format_version": 1,
        "d_model": 4,
        "total_tokens": 4,
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
    checkpoint = {
        "step": 25,
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "sae_state_dict": {"encoder.weight": torch.zeros(8, 4)},
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "format_version": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": 25,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_config_sha256": checkpoint["training_config_sha256"],
        "activation_manifest_sha256": checkpoint["activation_manifest_sha256"],
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer_index": config.model.layer_index,
        "features": [
            {
                "feature_id": 3,
                "activation_frequency": 0.01,
                "top_contexts": [
                    {"activation": 9.0, "text": "the bank approved a loan"},
                    {"activation": 8.0, "text": "credit was granted"},
                ],
                "random_active_contexts": [
                    {"activation": 4.0, "text": "the lender issued financing"}
                ],
                "negative_contexts": [
                    {"activation": 0.0, "text": "we sat on the river bank"},
                    {"activation": 0.0, "text": "the river flooded its banks"},
                ],
            }
        ],
    }
    report_path = run_dir / "feature_reports" / "features.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return config, checkpoint, checkpoint_path, manifest, report_path


def _provider_spec() -> ProviderSpec:
    return ProviderSpec(
        provider="transformers",
        model="fake",
        api_key_env="UNUSED",
        base_url=None,
        revision=None,
        device="cpu",
        dtype="float32",
        max_output_tokens=256,
        timeout_seconds=30,
        trust_remote_code=False,
    )


def test_label_registry_is_resumable_and_checkpoint_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, checkpoint, checkpoint_path, manifest, report_path = _label_fixture(tmp_path)
    fake = FakeJSONModel()
    monkeypatch.setattr(label_module, "build_json_model", lambda _spec: fake)

    destination = label_features(
        config,
        str(checkpoint_path),
        report_path=report_path,
        registry_path=None,
        feature_ids=None,
        provider_spec=_provider_spec(),
        scorer_spec=None,
        train_contexts=1,
        heldout_contexts=1,
        score=True,
        min_balanced_accuracy=0.7,
        min_spearman=0.4,
        retries=1,
        overwrite=False,
        acknowledge_external_data=False,
        dry_run=False,
    )

    identity = checkpoint_identity(config, checkpoint, checkpoint_path, manifest)
    registry = load_label_registry(destination, identity=identity)
    assert registry["labels"][0]["feature_id"] == 3
    assert registry["labels"][0]["status"] == "auto_validated"
    assert registry["labels"][0]["validation"]["balanced_accuracy"] == 1.0
    evidence = registry["labels"][0]["evidence"]
    assert evidence["heldout_text_in_registry"] is False
    assert evidence["local_snapshot_in_release"] is False
    assert (destination.parent / evidence["local_snapshot"]).exists()
    assert len(fake.calls) == 2

    label_features(
        config,
        str(checkpoint_path),
        report_path=report_path,
        registry_path=destination,
        feature_ids=None,
        provider_spec=_provider_spec(),
        scorer_spec=None,
        train_contexts=1,
        heldout_contexts=1,
        score=True,
        min_balanced_accuracy=0.7,
        min_spearman=0.4,
        retries=1,
        overwrite=False,
        acknowledge_external_data=False,
        dry_run=False,
    )
    assert len(fake.calls) == 2

    changed = {**identity, "checkpoint_step": 26}
    with pytest.raises(ValueError, match="different SAE checkpoint"):
        load_label_registry(destination, identity=changed)


def test_external_provider_requires_data_acknowledgement(
    tmp_path: Path,
) -> None:
    config, _, checkpoint_path, _, report_path = _label_fixture(tmp_path)
    external = ProviderSpec(
        **{
            **_provider_spec().__dict__,
            "provider": "openai",
            "model": "gpt-example",
            "api_key_env": "MISSING_TEST_KEY",
        }
    )
    with pytest.raises(RuntimeError, match="sends mined dataset text"):
        label_features(
            config,
            str(checkpoint_path),
            report_path=report_path,
            registry_path=None,
            feature_ids=None,
            provider_spec=external,
            scorer_spec=None,
            train_contexts=1,
            heldout_contexts=1,
            score=False,
            min_balanced_accuracy=0.7,
            min_spearman=0.4,
            retries=0,
            overwrite=False,
            acknowledge_external_data=False,
            dry_run=False,
        )


def test_openai_and_anthropic_use_structured_output_contracts(monkeypatch) -> None:
    calls = []

    def fake_post(url, *, headers, payload, timeout_seconds):
        calls.append((url, headers, payload, timeout_seconds))
        if url.endswith("/responses"):
            return {
                "id": "resp-1",
                "model": "gpt-example",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"ok": true}'}],
                    }
                ],
            }
        return {
            "id": "msg-1",
            "model": "claude-example",
            "content": [{"type": "text", "text": '{"ok": true}'}],
        }

    monkeypatch.setattr(label_module, "_post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    openai_model = OpenAIResponsesModel(
        "gpt-example",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        max_output_tokens=100,
        timeout_seconds=30,
    )
    anthropic_model = AnthropicMessagesModel(
        "claude-example",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        max_output_tokens=100,
        timeout_seconds=30,
    )
    assert openai_model.generate(
        system="system",
        prompt="prompt",
        schema=schema,
        schema_name="test",
    ).value == {"ok": True}
    assert anthropic_model.generate(
        system="system",
        prompt="prompt",
        schema=schema,
        schema_name="test",
    ).value == {"ok": True}
    assert calls[0][2]["text"]["format"]["strict"] is True
    assert calls[1][2]["output_config"]["format"]["type"] == "json_schema"


def test_openai_incomplete_response_exposes_reason_and_budget(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setattr(
        label_module,
        "_post_json",
        lambda *_args, **_kwargs: {
            "id": "resp-incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        },
    )
    model = OpenAIResponsesModel(
        "gpt-example",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        max_output_tokens=1024,
        timeout_seconds=30,
    )
    with pytest.raises(
        RetryableProviderResponseError,
        match=r"reason=max_output_tokens, max_output_tokens=1024",
    ):
        model.generate(
            system="system",
            prompt="prompt",
            schema={"type": "object"},
            schema_name="test",
        )


def test_validated_call_retries_incomplete_provider_response() -> None:
    class IncompleteOnceModel(label_module.JSONModel):
        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RetryableProviderResponseError("temporary incomplete response")
            return ModelResult({"ok": True}, {"response_id": "retry-success"})

    model = IncompleteOnceModel()
    value, metadata = label_module._call_validated(
        model,
        system="system",
        prompt="prompt",
        schema={"type": "object"},
        schema_name="test",
        validator=lambda result: result,
        retries=1,
    )
    assert value == {"ok": True}
    assert metadata["attempts"] == 2
    assert model.calls == 2


def test_openai_provider_uses_reasoning_safe_default_output_budget() -> None:
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--provider",
            "openai",
            "--model",
            "gpt-example",
        ]
    )
    assert provider_spec_from_args(args).max_output_tokens == 25_000


def test_corpus_evidence_keeps_development_and_validation_separate() -> None:
    feature = {
        "feature_id": 4,
        "activation_frequency": 0.1,
        "top_contexts": [
            {
                "activation": 4.0,
                "text": "development positive",
                "activating_token": "positive",
                "document_id": "development-positive",
                "evidence_split": "development",
            }
        ],
        "random_active_contexts": [
            {
                "activation": 2.0,
                "text": "validation positive",
                "activating_token": "positive",
                "document_id": "validation-positive",
                "evidence_split": "validation",
            }
        ],
        "negative_contexts": [
            {
                "activation": 0.0,
                "text": "development negative",
                "activating_token": "negative",
                "document_id": "development-negative",
                "evidence_split": "development",
            },
            {
                "activation": 0.0,
                "text": "validation negative",
                "activating_token": "negative",
                "document_id": "validation-negative",
                "evidence_split": "validation",
            },
        ],
    }
    training, heldout = label_module._feature_evidence(
        feature,
        train_contexts=1,
        heldout_contexts=1,
    )
    assert training["positive_examples"][0]["text"] == "development positive"
    assert training["positive_examples"][0]["target_token"] == "positive"
    assert training["zero_activation_examples"][0]["text"] == "development negative"
    assert training["zero_activation_examples"][0]["target_token"] == "negative"
    assert {example["text"] for example in heldout} == {
        "validation positive",
        "validation negative",
    }
    assert {example["target_token"] for example in heldout} == {
        "positive",
        "negative",
    }
