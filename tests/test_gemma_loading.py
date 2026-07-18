from __future__ import annotations

import torch
from torch import nn

from gemma4_sae import gemma as gemma_module
from gemma4_sae.config import ModelConfig


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def test_text_only_loader_uses_tokenizer_without_multimodal_processor(monkeypatch) -> None:
    tokenizer = object()
    model = TinyModel()
    tokenizer_call = {}
    model_call = {}

    def load_tokenizer(model_id, **kwargs):
        tokenizer_call.update(model_id=model_id, **kwargs)
        return tokenizer

    def load_model(model_id, **kwargs):
        model_call.update(model_id=model_id, **kwargs)
        return model

    monkeypatch.setattr(
        gemma_module.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        gemma_module.AutoModelForMultimodalLM,
        "from_pretrained",
        load_model,
    )
    monkeypatch.setattr(gemma_module, "read_hf_token", lambda: "test-token")
    monkeypatch.setattr(gemma_module, "select_device", lambda _backend: torch.device("cpu"))
    monkeypatch.setattr(
        gemma_module,
        "resolve_model_dtype",
        lambda _dtype, _device: torch.float32,
    )

    loaded_tokenizer, loaded_model = gemma_module.load_gemma(
        ModelConfig(
            revision="pinned-revision",
            backend="cpu",
            dtype="float32",
            device_map="auto",
        )
    )

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert tokenizer_call == {
        "model_id": "google/gemma-4-E4B",
        "revision": "pinned-revision",
        "token": "test-token",
    }
    assert model_call["model_id"] == "google/gemma-4-E4B"
    assert model_call["revision"] == "pinned-revision"
    assert model_call["token"] == "test-token"
    assert model_call["dtype"] == torch.float32
    assert model_call["low_cpu_mem_usage"] is True
    assert model_call["attn_implementation"] == "sdpa"
    assert "device_map" not in model_call
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
