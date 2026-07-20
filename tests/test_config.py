from pathlib import Path

import pytest

from gemma4_sae.config import load_config


def test_checked_in_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "e4b_layer20_batchtopk.yaml"
    config = load_config(path)
    assert config.model.model_id == "google/gemma-4-E4B"
    assert config.model.layer_index == 20
    assert len(config.model.revision) == 40
    assert len(config.data.revision) == 40
    assert config.sae.target_l0 == 64
    assert config.publication.hf_repo_id.startswith("lamm-mit/")


def test_instruction_tuned_config_uses_chat_messages() -> None:
    path = Path(__file__).parents[1] / "configs" / "e4b_it_layer20_batchtopk.yaml"
    config = load_config(path)
    assert config.model.model_id == "google/gemma-4-E4B-it"
    assert config.data.input_format == "messages"
    assert config.data.text_column == "messages"
    assert config.evaluation.split == "test_sft"


def test_dgx_spark_config_is_full_scale_cuda_run() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "e4b_layer20_batchtopk_dgx_spark.yaml"
    )
    config = load_config(path)
    assert config.model.backend == "cuda"
    assert config.model.dtype == "bfloat16"
    assert config.model.load_in_4bit is False
    assert config.data.max_activation_tokens == 50_000_000
    assert config.sae.expansion_factor == 16
    assert config.sae.train_batch_size == 4_096
    assert config.sae.max_steps == 25_000
    assert config.sae.auxiliary_loss_coefficient == 1 / 32
    assert config.sae.auxiliary_top_k == 512
    assert config.sae.dead_after_steps == 5
    assert config.sae.resample_dead_features is False
    assert config.evaluation.max_sequences == 256


def test_selected_12x_config_publishes_portable_example() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "e4b_layer20_batchtopk_dgx_spark_12x_l064.yaml"
    )
    config = load_config(path)
    assert config.sae.expansion_factor == 12
    assert (
        config.publication.hf_repo_id
        == "lamm-mit/gemma-4-e4b-layer20-batchtopk-sae"
    )
    assert (
        config.publication.example_explanation_path
        == "runs/hub-smoke-test-12x.json"
    )


def test_unknown_configuration_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model:
  model_id: google/gemma-4-E4B
  mystery: true
data: {}
sae: {}
evaluation: {}
publication: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown ModelConfig fields"):
        load_config(path)
