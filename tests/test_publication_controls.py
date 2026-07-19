import pytest

from gemma4_sae.checkpoint import validate_checkpoint_provenance
from gemma4_sae.cli import build_parser
from gemma4_sae.fidelity import bootstrap_intervals
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


def test_unified_cli_has_all_pipeline_commands() -> None:
    parser = build_parser()
    for command in (
        "doctor",
        "collect",
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
        namespace = parser.parse_args(arguments)
        assert namespace.command == command
