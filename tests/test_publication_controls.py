from types import SimpleNamespace

import pytest
import torch

import gemma4_sae.fidelity as fidelity_module
from gemma4_sae.checkpoint import validate_checkpoint_provenance
from gemma4_sae.cli import build_parser
from gemma4_sae.fidelity import bootstrap_intervals, mean_language_model_loss
from gemma4_sae.provenance import canonical_sha256, training_config_sha256


def test_checkpoint_provenance_rejects_changed_manifest() -> None:
    config = {
        "model": {"model_id": "example"},
        "data": {},
        "sae": {},
        "evaluation": {"max_sequences": 8},
    }
    manifest = {"total_tokens": 10}
    checkpoint = {
        "training_config_sha256": training_config_sha256(config),
        "activation_manifest_sha256": canonical_sha256(manifest),
    }
    validate_checkpoint_provenance(checkpoint, config, manifest)
    changed_evaluation = {**config, "evaluation": {"max_sequences": 64}}
    validate_checkpoint_provenance(checkpoint, changed_evaluation, manifest)
    with pytest.raises(ValueError, match="activation manifest"):
        validate_checkpoint_provenance(
            checkpoint,
            config,
            {"total_tokens": 11},
        )


def test_bootstrap_intervals_are_deterministic() -> None:
    first = bootstrap_intervals(
        [2.0, 2.1, 1.9],
        [2.05, 2.15, 1.95],
        [3.0, 3.1, 2.9],
        seed=17,
        samples=100,
    )
    second = bootstrap_intervals(
        [2.0, 2.1, 1.9],
        [2.05, 2.15, 1.95],
        [3.0, 3.1, 2.9],
        seed=17,
        samples=100,
    )
    assert first == second
    assert len(first["loss_recovered_95ci"]) == 2


def test_fidelity_loss_reports_progress(monkeypatch) -> None:
    progress: dict[str, object] = {}

    def fake_tqdm(iterable, **kwargs):
        progress.update(kwargs)
        return iterable

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids, **_kwargs):
            return SimpleNamespace(loss=input_ids.float().mean() / 10.0)

    monkeypatch.setattr(fidelity_module, "tqdm", fake_tqdm)
    loss, predicted_tokens, sequence_losses = mean_language_model_loss(
        ToyModel(),
        [torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])],
        batch_size=1,
        progress_description="Fidelity test",
    )

    assert loss == pytest.approx(0.25)
    assert predicted_tokens == 4
    assert sequence_losses == pytest.approx([0.2, 0.3])
    assert progress == {
        "total": 2,
        "desc": "Fidelity test",
        "unit": "batch",
        "dynamic_ncols": True,
    }


def test_unified_cli_has_all_pipeline_commands() -> None:
    parser = build_parser()
    for command in (
        "doctor",
        "collect",
        "develop-labels",
        "verify",
        "train",
        "evaluate",
        "explain",
        "label",
        "fidelity",
        "mine",
        "publish",
    ):
        arguments = [command, "--config", "config.yaml"]
        if command == "explain":
            arguments.extend(["--text", "hello"])
        if command == "label":
            arguments.extend(["--provider", "transformers", "--model", "test/model"])
        if command == "develop-labels":
            arguments.extend(
                [
                    "--corpus",
                    "corpus.jsonl",
                    "--provider",
                    "transformers",
                    "--model",
                    "test/model",
                ]
            )
        namespace = parser.parse_args(arguments)
        assert namespace.command == command

    hub_explain = parser.parse_args(
        [
            "explain",
            "--sae-repo",
            "lamm-mit/test-sae",
            "--device",
            "cpu",
            "--text",
            "hello",
        ]
    )
    assert hub_explain.config is None
    assert hub_explain.sae_repo == "lamm-mit/test-sae"
    assert hub_explain.device == "cpu"
