from __future__ import annotations

import math
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .benchmark import load_normalization_from_checkpoint
from .data import DEFAULT_PARTICLE_FEATURES
from .model import masked_mean


def _project_last_dim(values: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    projected = torch.matmul(values, weight.transpose(0, 1))
    if bias is not None:
        projected = projected + bias
    return projected


class _ExportableSelfAttention(nn.Module):
    """Self-attention with MultiheadAttention-compatible parameter names."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, token_count, _ = x.shape

        qkv = _project_last_dim(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q, batch_size, token_count)
        k = self._split_heads(k, batch_size, token_count)
        v = self._split_heads(v, batch_size, token_count)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            invalid_keys = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, -1.0e9)

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            token_count,
            self.embed_dim,
        )
        return _project_last_dim(
            context,
            self.out_proj.weight,
            self.out_proj.bias,
        )

    def _split_heads(self, values: Tensor, batch_size: int, token_count: int) -> Tensor:
        return values.view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)


class _BenchmarkExportSimpleParT(nn.Module):
    """SimpleParT variant that exports to a batch-friendly ONNX graph."""

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
        self.attention = _ExportableSelfAttention(
            embed_dim=d_model,
            num_heads=num_heads,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_particles: Tensor, padding_mask: Tensor) -> Tensor:
        x = _project_last_dim(
            x_particles,
            self.embedding.weight,
            self.embedding.bias,
        )
        attention_padding_mask = ~padding_mask
        attended = self.attention(
            x,
            key_padding_mask=attention_padding_mask,
        )
        pooled = masked_mean(attended, padding_mask)
        return _project_last_dim(
            pooled,
            self.classifier.weight,
            self.classifier.bias,
        )


class _VisualExportSelfAttention(nn.Module):
    """Checkpoint-compatible attention with a cleaner high-level ONNX export path."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, token_count, _ = x.shape

        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            token_count,
            self.embed_dim,
        )
        return self.out_proj(context)


class _VisualExportSimpleParT(nn.Module):
    """SimpleParT variant aimed at a cleaner visualization-oriented ONNX graph."""

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
        self.attention = _VisualExportSelfAttention(
            embed_dim=d_model,
            num_heads=num_heads,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_particles: Tensor, padding_mask: Tensor) -> Tensor:
        x = self.embedding(x_particles)
        attention_padding_mask = ~padding_mask
        attended = self.attention(
            x,
            key_padding_mask=attention_padding_mask,
        )
        pooled = masked_mean(attended, padding_mask)
        return self.classifier(pooled)


class ExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        normalization_mean: torch.Tensor | None,
        normalization_std: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        if normalization_mean is not None and normalization_std is not None:
            self.register_buffer("normalization_mean", normalization_mean)
            self.register_buffer("normalization_std", normalization_std)
        else:
            self.normalization_mean = None
            self.normalization_std = None

    def forward(self, x_particles: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if self.normalization_mean is not None and self.normalization_std is not None:
            x_particles = (x_particles - self.normalization_mean) / self.normalization_std
            x_particles = torch.where(
                padding_mask.unsqueeze(-1),
                x_particles,
                torch.zeros_like(x_particles),
            )
        return self.model(x_particles, padding_mask)


class SoftmaxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_particles: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(x_particles, padding_mask)
        return torch.softmax(logits, dim=1)


def build_export_model(*, variant: str) -> nn.Module:
    input_dim = len(DEFAULT_PARTICLE_FEATURES)
    if variant == "benchmark":
        return _BenchmarkExportSimpleParT(input_dim=input_dim)
    if variant == "visual":
        return _VisualExportSimpleParT(input_dim=input_dim)
    raise ValueError(f"Unsupported export variant: {variant!r}")


def resolve_output_kind(variant: str) -> str:
    if variant == "visual":
        return "probabilities"
    return "logits"


def resolve_output_name(variant: str) -> str:
    if variant == "visual":
        return "probabilities"
    return "logits"


def attach_metadata(
    output_path: Path,
    *,
    variant: str,
    output_kind: str,
    has_embedded_normalization: bool,
    max_constituents: int,
    dynamic_batch: bool,
) -> None:
    model = onnx.load(str(output_path), load_external_data=False)
    metadata = {
        "jet_tagger_model": "SimpleParT",
        "jet_tagger_output_kind": output_kind,
        "jet_tagger_embedded_normalization": "true" if has_embedded_normalization else "false",
        "jet_tagger_particle_features": ",".join(DEFAULT_PARTICLE_FEATURES),
        "jet_tagger_max_constituents": str(max_constituents),
        "jet_tagger_export_variant": variant,
        "jet_tagger_dynamic_batch": "true" if dynamic_batch else "false",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(output_path))


def export_checkpoint_to_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    max_constituents: int,
    opset: int,
    variant: str,
    embed_normalization: bool,
    dynamic_batch: bool,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    base_model = build_export_model(variant=variant)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    output_kind = resolve_output_kind(variant)
    output_name = resolve_output_name(variant)

    normalization = load_normalization_from_checkpoint(checkpoint)
    normalization_mean = None
    normalization_std = None
    if embed_normalization and normalization is not None:
        normalization_mean = torch.as_tensor(
            normalization.mean,
            dtype=torch.float32,
        ).view(1, 1, -1)
        normalization_std = torch.as_tensor(
            normalization.std,
            dtype=torch.float32,
        ).view(1, 1, -1)

    model = ExportWrapper(base_model, normalization_mean, normalization_std)
    if variant == "visual":
        model = SoftmaxWrapper(model)
    model.eval()

    x_particles = torch.randn(
        1,
        max_constituents,
        len(DEFAULT_PARTICLE_FEATURES),
        dtype=torch.float32,
    )
    padding_mask = torch.ones(
        1,
        max_constituents,
        dtype=torch.bool,
    )

    export_kwargs = {
        "dynamo": False,
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["x_particles", "padding_mask"],
        "output_names": [output_name],
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "x_particles": {0: "batch_size"},
            "padding_mask": {0: "batch_size"},
            output_name: {0: "batch_size"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (x_particles, padding_mask),
            output_path,
            **export_kwargs,
        )

    attach_metadata(
        output_path,
        variant=variant,
        output_kind=output_kind,
        has_embedded_normalization=normalization is not None and embed_normalization,
        max_constituents=max_constituents,
        dynamic_batch=dynamic_batch,
    )
