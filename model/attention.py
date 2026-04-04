"""Attention layers used by the transformer."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Create a causal mask for autoregressive decoding."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    The score for each query-key pair is:

    attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attention_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit query, key, and value projections."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.size()
        tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.d_head)
        return tensor.transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, sequence_length, d_head = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, sequence_length, num_heads * d_head)
        return tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention to a batch of sequences."""
        projected_query = self._split_heads(self.query_projection(query))
        projected_key = self._split_heads(self.key_projection(key))
        projected_value = self._split_heads(self.value_projection(value))
        attended, weights = scaled_dot_product_attention(
            projected_query,
            projected_key,
            projected_value,
            mask=mask,
            dropout=self.dropout,
        )
        combined = self._combine_heads(attended)
        output = self.output_projection(combined)
        return output, weights


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network used inside transformer blocks."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project inputs up to the feed-forward width and back down."""
        hidden = self.linear1(inputs)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.linear2(hidden)
