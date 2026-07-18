from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from .config import ModelConfig
from .devices import resolve_model_dtype, select_device


def read_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def load_gemma(config: ModelConfig):
    token = read_hf_token()
    device = select_device(config.backend)
    dtype = resolve_model_dtype(config.dtype, device)
    # Activation collection and fidelity evaluation are text-only. Loading Gemma 4's
    # multimodal AutoProcessor would import torchvision even though no image is ever
    # supplied, so load the checkpoint tokenizer directly.
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        token=token,
    )
    load_kwargs = {
        "token": token,
        "revision": config.revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    # Accelerate device maps are useful for CUDA sharding/offload. A direct move is
    # more predictable for MPS and CPU.
    if device.type == "cuda" and config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit bitsandbytes loading is supported only on CUDA.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "4-bit loading requires the optional dependency: "
                "pip install -e '.[quantization]'"
            )
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForMultimodalLM.from_pretrained(config.model_id, **load_kwargs)
    if device.type in {"mps", "cpu"}:
        model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def find_text_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    text_config = model.config.get_text_config()
    expected = text_config.num_hidden_layers
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected:
            if any("DecoderLayer" in child.__class__.__name__ for child in module):
                candidates.append((name, module))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(f"Expected one Gemma text decoder stack; found {names}.")
    return candidates[0]


class GemmaActivationExtractor:
    """Capture one Gemma 4 decoder-layer output with a forward hook."""

    def __init__(self, model: nn.Module, layer_index: int) -> None:
        self.model = model
        self.layer_path, layers = find_text_decoder_layers(model)
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer_index must be in [0, {len(layers) - 1}].")
        self.layer_index = layer_index
        self.layer = layers[layer_index]
        self.input_device = model.get_input_embeddings().weight.device
        self._captured: Tensor | None = None
        self._handle = None

    def __enter__(self) -> GemmaActivationExtractor:
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            self._captured = value.detach()

        self._handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.inference_mode()
    def extract(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if self._handle is None:
            raise RuntimeError("Use GemmaActivationExtractor as a context manager.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        self._captured = None
        inputs = {
            "input_ids": input_ids.to(self.input_device),
            "attention_mask": attention_mask.to(self.input_device),
        }
        outputs = self.model(
            **inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        del outputs
        if self._captured is None:
            raise RuntimeError("The Gemma layer hook did not fire.")
        result = self._captured
        self._captured = None
        return result


def contexts_for_tokens(input_ids: Tensor, radius: int, pad_token_id: int) -> Tensor:
    """Return [batch, sequence, 2 * radius + 1] token windows."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    width = 2 * radius + 1
    padded = F.pad(input_ids, (radius, radius), value=pad_token_id)
    return padded.unfold(dimension=1, size=width, step=1)


def valid_token_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    special_token_ids: Sequence[int],
) -> Tensor:
    mask = attention_mask.bool()
    for token_id in special_token_ids:
        mask &= input_ids != token_id
    return mask
