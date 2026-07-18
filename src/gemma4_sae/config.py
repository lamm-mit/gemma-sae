from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "google/gemma-4-E4B"
    layer_index: int = 20
    dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    sequence_length: int = 512
    inference_batch_size: int = 2


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "HuggingFaceFW/fineweb"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    shuffle_buffer: int = 10_000
    min_chars: int = 200
    max_activation_tokens: int = 5_000_000
    tokens_per_shard: int = 65_536
    context_radius: int = 12
    activation_dir: str = "activations/gemma-4-e4b/layer-20"
    seed: int = 17


@dataclass(frozen=True)
class SAEConfig:
    architecture: str = "batchtopk"
    expansion_factor: int = 8
    target_l0: int = 64
    train_batch_size: int = 512
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    max_steps: int = 50_000
    warmup_steps: int = 1_000
    gradient_clip_norm: float = 1.0
    threshold_ema_decay: float = 0.999
    dead_after_steps: int = 2_500
    resample_every_steps: int = 2_500
    max_resamples_per_event: int = 512
    checkpoint_every_steps: int = 2_500
    log_every_steps: int = 25
    validation_fraction: float = 0.02
    run_dir: str = "runs/e4b-layer20-batchtopk"
    seed: int = 17


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    sae: SAEConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.model.model_id.lower() not in {
            "google/gemma-4-e4b",
            "google/gemma-4-e4b-it",
        }:
            raise ValueError("This repository currently targets Gemma 4 E4B checkpoints.")
        if self.model.layer_index < 0:
            raise ValueError("model.layer_index must be non-negative.")
        if self.model.sequence_length < 8:
            raise ValueError("model.sequence_length must be at least 8.")
        if self.data.max_activation_tokens <= 0:
            raise ValueError("data.max_activation_tokens must be positive.")
        if self.data.tokens_per_shard <= 0:
            raise ValueError("data.tokens_per_shard must be positive.")
        if self.data.context_radius < 0:
            raise ValueError("data.context_radius must be non-negative.")
        if self.sae.architecture != "batchtopk":
            raise ValueError("Only the batchtopk architecture is implemented.")
        if self.sae.expansion_factor < 1:
            raise ValueError("sae.expansion_factor must be at least 1.")
        if self.sae.target_l0 < 1:
            raise ValueError("sae.target_l0 must be positive.")
        if not 0 <= self.sae.validation_fraction < 0.5:
            raise ValueError("sae.validation_fraction must be in [0, 0.5).")


def _strict_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    required = {"model", "data", "sae"}
    missing = required - set(raw)
    unknown = set(raw) - required
    if missing or unknown:
        raise ValueError(
            f"Configuration sections missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    config = ProjectConfig(
        model=_strict_dataclass(ModelConfig, raw["model"]),
        data=_strict_dataclass(DataConfig, raw["data"]),
        sae=_strict_dataclass(SAEConfig, raw["sae"]),
    )
    config.validate()
    return config
