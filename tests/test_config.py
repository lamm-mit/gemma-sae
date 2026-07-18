from pathlib import Path

import pytest

from gemma4_sae.config import load_config


def test_checked_in_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "e4b_layer20_batchtopk.yaml"
    config = load_config(path)
    assert config.model.model_id == "google/gemma-4-E4B"
    assert config.model.layer_index == 20
    assert config.sae.target_l0 == 64


def test_unknown_configuration_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model:
  model_id: google/gemma-4-E4B
  mystery: true
data: {}
sae: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown ModelConfig fields"):
        load_config(path)

