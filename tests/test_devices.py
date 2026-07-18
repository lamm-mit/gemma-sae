import pytest
import torch

from gemma4_sae.devices import resolve_model_dtype, select_device


def test_cpu_auto_dtype_is_float32() -> None:
    assert resolve_model_dtype("auto", torch.device("cpu")) == torch.float32
    assert select_device("cpu") == torch.device("cpu")


def test_cpu_float16_is_rejected() -> None:
    with pytest.raises(ValueError, match="float16 model execution on CPU"):
        resolve_model_dtype("float16", torch.device("cpu"))


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        select_device("quantum")
