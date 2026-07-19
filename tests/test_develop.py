from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

import gemma4_sae.develop as develop_module
from gemma4_sae.config import DataConfig, ModelConfig, ProjectConfig, SAEConfig
from gemma4_sae.develop import develop_labels, load_local_corpus, select_corpus_features
from gemma4_sae.provenance import canonical_sha256, training_config_sha256


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    all_special_ids = [0]

    def __call__(self, text, **_kwargs):
        return {"input_ids": [(len(word) % 3) + 1 for word in text.split()]}

    def decode(self, token_ids, **_kwargs):
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class FakeExtractor:
    def __init__(self, _model, _layer_index):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        pass

    def extract(self, input_ids, _attention_mask):
        hidden = torch.zeros(*input_ids.shape, 4)
        for index in range(1, 4):
            hidden[..., index][input_ids == index] = float(index)
        return hidden


class FakeSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.n_features = 4

    def encode(self, x, use_threshold=True):
        del use_threshold
        features = torch.relu(x)
        selected = features.nonzero(as_tuple=False)[:, -1]
        return features, selected, torch.tensor(0.0)


def _fixture(tmp_path: Path):
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
        model=ModelConfig(backend="cpu", dtype="float32", inference_batch_size=2),
        data=DataConfig(activation_dir=str(activation_dir)),
        sae=SAEConfig(expansion_factor=1, target_l0=1, run_dir=str(run_dir)),
    )
    checkpoint = {
        "step": 10,
        "training_config_sha256": training_config_sha256(config.to_dict()),
        "activation_manifest_sha256": canonical_sha256(manifest),
        "sae_state_dict": {"encoder.weight": torch.zeros(4, 4)},
        "activation_mean": torch.zeros(4),
        "activation_scale": torch.tensor(1.0),
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {"id": "d1", "split": "development", "text": "alpha beta gamma delta"},
        {"id": "d2", "split": "development", "text": "physics atoms lattice model"},
        {"id": "v1", "split": "validation", "text": "held out atoms control"},
        {"id": "v2", "split": "validation", "text": "held out lattice test"},
    ]
    corpus_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return config, checkpoint_path, corpus_path


def test_load_local_corpus_supports_explicit_and_generated_splits(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text(
        '{"id":"d","split":"development","text":"one"}\n'
        '{"id":"v","split":"validation","text":"two"}\n',
        encoding="utf-8",
    )
    documents, provenance = load_local_corpus(
        explicit,
        text_column="text",
        input_format="text",
        id_column="id",
        split_column="split",
        development_split="development",
        validation_split="validation",
        validation_fraction=0.2,
        seed=17,
    )
    assert [document.split for document in documents] == ["development", "validation"]
    assert provenance["split_method"] == "explicit_column"

    generated = tmp_path / "generated.txt"
    generated.write_text("one record\n\ntwo record\n\nthree record\n", encoding="utf-8")
    first, first_provenance = load_local_corpus(
        generated,
        text_column="text",
        input_format="text",
        id_column="id",
        split_column="split",
        development_split="development",
        validation_split="validation",
        validation_fraction=0.2,
        seed=17,
    )
    second, _ = load_local_corpus(
        generated,
        text_column="text",
        input_format="text",
        id_column="id",
        split_column="split",
        development_split="development",
        validation_split="validation",
        validation_fraction=0.2,
        seed=17,
    )
    assert [document.split for document in first] == [
        document.split for document in second
    ]
    assert {document.split for document in first} == {"development", "validation"}
    assert first_provenance["split_method"] == "deterministic_hash"


def test_select_corpus_features_records_auditable_metrics() -> None:
    selected, metrics, formula = select_corpus_features(
        {
            "counts": torch.tensor([0.0, 8.0, 3.0, 5.0]),
            "strengths": torch.tensor([0.0, 16.0, 15.0, 5.0]),
            "document_counts": torch.tensor([0.0, 4.0, 2.0, 5.0]),
            "token_count": 20,
            "document_count": 5,
        },
        n_features=2,
        ranking="coverage",
        min_active_contexts=2,
        min_document_frequency=0.1,
        max_document_frequency=1.0,
    )
    assert selected == [2, 1]
    assert metrics[2]["selection_rank"] == 1
    assert metrics[1]["active_document_count"] == 4
    assert "document_frequency" in formula


def test_develop_labels_writes_split_safe_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, checkpoint_path, corpus_path = _fixture(tmp_path)
    captured = {}
    monkeypatch.setattr(
        develop_module,
        "load_gemma",
        lambda _config: (FakeTokenizer(), object()),
    )
    monkeypatch.setattr(develop_module, "GemmaActivationExtractor", FakeExtractor)
    monkeypatch.setattr(
        develop_module,
        "build_sae_from_checkpoint",
        lambda _checkpoint, _device: FakeSAE(),
    )
    monkeypatch.setattr(
        develop_module,
        "select_device",
        lambda _backend: torch.device("cpu"),
    )

    def fake_label_features(_config, _checkpoint_request, **kwargs):
        captured.update(kwargs)
        return Path(_config.sae.run_dir) / "feature_labels" / "labels.json"

    monkeypatch.setattr(develop_module, "label_features", fake_label_features)
    report_path, registry_path = develop_labels(
        config,
        str(checkpoint_path),
        corpus_path=corpus_path,
        report_path=None,
        text_column="text",
        input_format="text",
        id_column="id",
        split_column="split",
        development_split="development",
        validation_split="validation",
        validation_fraction=0.2,
        max_documents=None,
        max_tokens=32,
        context_radius=2,
        n_features=2,
        ranking="coverage",
        min_active_contexts=1,
        min_document_frequency=0.0,
        max_document_frequency=1.0,
        provider_spec=object(),
        scorer_spec=None,
        registry_path=None,
        train_contexts=1,
        heldout_contexts=1,
        score=True,
        min_balanced_accuracy=0.7,
        min_spearman=0.4,
        retries=1,
        overwrite=False,
        acknowledge_external_data=False,
        dry_run=True,
    )

    report = json.loads(report_path.read_text())
    assert registry_path.name == "labels.json"
    assert report["corpus_report_version"] == 1
    assert report["selection_policy"]["selection_uses_heldout_split"] is False
    assert report["corpus"]["development_records"] == 2
    assert report["corpus"]["validation_records"] == 2
    assert captured["report_path"] == report_path
    for feature in report["features"]:
        assert {
            item["evidence_split"] for item in feature["top_contexts"]
        } <= {"development"}
        assert {
            item["evidence_split"] for item in feature["random_active_contexts"]
        } <= {"validation"}
