from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SAEOutput:
    reconstruction: Tensor
    features: Tensor
    selected_indices: Tensor
    batch_threshold: Tensor


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE with thresholded inference and unit-norm decoder features."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        target_l0: int,
        threshold_ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_features <= 0:
            raise ValueError("d_model and n_features must be positive.")
        if not 1 <= target_l0 <= n_features:
            raise ValueError("target_l0 must be between 1 and n_features.")
        if not 0.0 <= threshold_ema_decay < 1.0:
            raise ValueError("threshold_ema_decay must be in [0, 1).")

        self.d_model = d_model
        self.n_features = n_features
        self.target_l0 = target_l0
        self.threshold_ema_decay = threshold_ema_decay

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("inference_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_initialized", torch.tensor(False))

        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder_()

    def preactivations(self, x: Tensor) -> Tensor:
        return F.relu(self.encoder(x - self.decoder_bias))

    def _batch_topk(self, dense: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = dense.flatten()
        requested = dense.shape[0] * self.target_l0
        k = min(requested, flat.numel())
        values, indices = torch.topk(flat, k=k, sorted=False)
        sparse = torch.zeros_like(flat).scatter(0, indices, values).view_as(dense)
        positive = values > 0
        threshold = values[positive].min().detach() if positive.any() else values.new_zeros(())
        feature_indices = indices[positive].remainder(self.n_features)
        return sparse, feature_indices, threshold

    def _threshold_encode(self, dense: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        threshold = self.inference_threshold.to(dtype=dense.dtype)
        sparse = dense * (dense > threshold)
        selected = sparse.nonzero(as_tuple=False)[:, -1]
        return sparse, selected, threshold

    @torch.no_grad()
    def update_inference_threshold_(self, batch_threshold: Tensor) -> None:
        value = batch_threshold.to(self.inference_threshold)
        if not bool(self.threshold_initialized):
            self.inference_threshold.copy_(value)
            self.threshold_initialized.fill_(True)
        else:
            decay = self.threshold_ema_decay
            self.inference_threshold.mul_(decay).add_(value, alpha=1.0 - decay)

    def encode(self, x: Tensor, use_threshold: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        dense = self.preactivations(x)
        if use_threshold is None:
            use_threshold = not self.training
        if use_threshold:
            return self._threshold_encode(dense)
        return self._batch_topk(dense)

    def decode(self, features: Tensor) -> Tensor:
        return self.decoder(features) + self.decoder_bias

    def forward(self, x: Tensor, use_threshold: bool | None = None) -> SAEOutput:
        features, selected, threshold = self.encode(x, use_threshold=use_threshold)
        return SAEOutput(
            reconstruction=self.decode(features),
            features=features,
            selected_indices=selected,
            batch_threshold=threshold,
        )

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def remove_decoder_gradient_parallel_component_(self) -> None:
        gradient = self.decoder.weight.grad
        if gradient is None:
            return
        directions = self.decoder.weight
        parallel = (gradient * directions).sum(dim=0, keepdim=True)
        gradient.sub_(parallel * directions)

    @torch.no_grad()
    def resample_dead_features_(
        self,
        dead_indices: Tensor,
        residual: Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        if dead_indices.numel() == 0:
            return 0
        dead_indices = dead_indices.to(device=self.decoder.weight.device, dtype=torch.long)
        residual = residual.detach().to(self.decoder.weight.device)
        probabilities = residual.square().sum(dim=-1)
        if float(probabilities.sum()) == 0.0:
            probabilities = torch.ones_like(probabilities)
        sample_rows = torch.multinomial(
            probabilities,
            num_samples=dead_indices.numel(),
            replacement=dead_indices.numel() > residual.shape[0],
        )
        directions = F.normalize(residual[sample_rows], dim=-1)

        self.decoder.weight[:, dead_indices] = directions.T
        self.encoder.weight[dead_indices] = directions * 0.2
        self.encoder.bias[dead_indices] = 0.0
        self._clear_optimizer_state(optimizer, dead_indices)
        return dead_indices.numel()

    @torch.no_grad()
    def _clear_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer | None,
        feature_indices: Tensor,
    ) -> None:
        if optimizer is None:
            return
        feature_indices = feature_indices.long()
        for parameter, axis in (
            (self.encoder.weight, 0),
            (self.encoder.bias, 0),
            (self.decoder.weight, 1),
        ):
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                index = [slice(None)] * value.ndim
                index[axis] = feature_indices
                value[tuple(index)] = 0
