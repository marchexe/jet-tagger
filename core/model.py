from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleParT(nn.Module):
    """A minimal ParT-inspired classifier for fixed-size particle tokens."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        d_model: int = 64,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.embedding = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_particles: Tensor, padding_mask: Tensor) -> Tensor:
        x = self.embedding(x_particles)
        attended, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=~padding_mask,
            need_weights=False,
        )
        pooled = masked_mean(attended, padding_mask)
        return self.classifier(pooled)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    total = (values * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return total / counts
