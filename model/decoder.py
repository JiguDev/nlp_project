"""Transformer decoder blocks."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model.attention import MultiHeadAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Single transformer decoder block with masked self-attention and cross-attention."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run masked self-attention, cross-attention, and the feed-forward network."""
        self_attn_output, _ = self.self_attention(inputs, inputs, inputs, mask=tgt_mask)
        inputs = self.norm1(inputs + self.dropout(self_attn_output))
        cross_attn_output, _ = self.cross_attention(inputs, encoder_outputs, encoder_outputs, mask=memory_mask)
        inputs = self.norm2(inputs + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(inputs)
        outputs = self.norm3(inputs + self.dropout(ff_output))
        return outputs


class Decoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(self, layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if index == 0 else type(layer)(
            layer.self_attention.d_model,
            layer.self_attention.num_heads,
            layer.feed_forward.linear1.out_features,
            layer.dropout.p,
        ) for index in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attention.d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the decoder stack to a target prefix."""
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, encoder_outputs, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(outputs)
