"""Transformer encoder blocks."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model.attention import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Single transformer encoder block with residual connections and layer norm."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run self-attention followed by the feed-forward network."""
        attn_output, _ = self.self_attention(inputs, inputs, inputs, mask=src_mask)
        inputs = self.norm1(inputs + self.dropout(attn_output))
        ff_output = self.feed_forward(inputs)
        outputs = self.norm2(inputs + self.dropout(ff_output))
        return outputs


class Encoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if index == 0 else type(layer)(
            layer.self_attention.d_model,
            layer.self_attention.num_heads,
            layer.feed_forward.linear1.out_features,
            layer.dropout.p,
        ) for index in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attention.d_model)

    def forward(self, inputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the encoder stack to the source sequence."""
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, src_mask=src_mask)
        return self.norm(outputs)
