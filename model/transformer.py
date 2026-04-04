"""Transformer sequence-to-sequence model implemented from scratch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from model.attention import subsequent_mask
from model.decoder import Decoder, DecoderLayer
from model.encoder import Encoder, EncoderLayer


@dataclass
class TransformerConfig:
    """Hyperparameters for the transformer architecture."""

    vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 256
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    tie_embeddings: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in the original transformer paper."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add positional information to token embeddings."""
        length = inputs.size(1)
        return self.dropout(inputs + self.pe[:, :length])


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder transformer for multilingual text generation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_scale = config.d_model ** 0.5
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position_encoding = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        encoder_layer = EncoderLayer(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
        decoder_layer = DecoderLayer(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
        self.encoder = Encoder(encoder_layer, config.num_encoder_layers)
        self.decoder = Decoder(decoder_layer, config.num_decoder_layers)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and add positional encoding."""
        return self.position_encoding(self.token_embedding(tokens) * self.embedding_scale)

    def _make_source_mask(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """Create a padding mask for the source sequence."""
        return src_tokens.eq(self.config.pad_id)

    def _make_target_mask(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Create a combined padding and causal mask for the decoder input."""
        padding_mask = tgt_tokens.eq(self.config.pad_id)
        causal_mask = subsequent_mask(tgt_tokens.size(1), device=tgt_tokens.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        return causal_mask | padding_mask

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Run the full encoder-decoder transformer."""
        src_mask = self._make_source_mask(src_tokens).unsqueeze(1).unsqueeze(2)
        tgt_mask = self._make_target_mask(tgt_tokens)
        src_embeddings = self._embed(src_tokens)
        tgt_embeddings = self._embed(tgt_tokens)
        encoder_outputs = self.encoder(src_embeddings, src_mask=src_mask)
        decoder_outputs = self.decoder(tgt_embeddings, encoder_outputs, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.output_projection(decoder_outputs)

    def encode(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """Encode a source sequence for downstream decoding."""
        src_mask = self._make_source_mask(src_tokens).unsqueeze(1).unsqueeze(2)
        return self.encoder(self._embed(src_tokens), src_mask=src_mask)

    def decode_step(self, src_tokens: torch.Tensor, generated_tokens: torch.Tensor) -> torch.Tensor:
        """Produce logits for the next token during autoregressive decoding."""
        src_mask = self._make_source_mask(src_tokens).unsqueeze(1).unsqueeze(2)
        tgt_mask = self._make_target_mask(generated_tokens)
        encoder_outputs = self.encoder(self._embed(src_tokens), src_mask=src_mask)
        decoder_outputs = self.decoder(self._embed(generated_tokens), encoder_outputs, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.output_projection(decoder_outputs)

    @torch.no_grad()
    def greedy_decode(self, src_tokens: torch.Tensor, max_new_tokens: int = 80) -> torch.Tensor:
        """Generate a response with greedy decoding."""
        self.eval()
        generated = torch.full((src_tokens.size(0), 1), self.config.bos_id, dtype=torch.long, device=src_tokens.device)
        for _ in range(max_new_tokens):
            logits = self.decode_step(src_tokens, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.eq(self.config.eos_id)):
                break
        return generated

    @torch.no_grad()
    def sample_decode(
        self,
        src_tokens: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate a response with temperature sampling and optional top-k filtering."""
        self.eval()
        generated = torch.full((src_tokens.size(0), 1), self.config.bos_id, dtype=torch.long, device=src_tokens.device)
        for _ in range(max_new_tokens):
            logits = self.decode_step(src_tokens, generated)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.eq(self.config.eos_id)):
                break
        return generated

    @torch.no_grad()
    def beam_search_decode(self, src_tokens: torch.Tensor, max_new_tokens: int = 80, beam_width: int = 4) -> torch.Tensor:
        """Generate a response using beam search."""
        self.eval()
        batch_size = src_tokens.size(0)
        if batch_size != 1:
            raise ValueError("beam_search_decode currently supports batch_size=1")
        beams = [(torch.tensor([[self.config.bos_id]], device=src_tokens.device), 0.0)]
        for _ in range(max_new_tokens):
            candidates = []
            for tokens, score in beams:
                if tokens[0, -1].item() == self.config.eos_id:
                    candidates.append((tokens, score))
                    continue
                logits = self.decode_step(src_tokens, tokens)[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                top_scores, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                for token_score, token_id in zip(top_scores[0], top_indices[0]):
                    candidate_tokens = torch.cat([tokens, token_id.view(1, 1)], dim=1)
                    candidates.append((candidate_tokens, score + float(token_score)))
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
            if all(tokens[0, -1].item() == self.config.eos_id for tokens, _ in beams):
                break
        return beams[0][0]

    def export_config(self) -> dict:
        """Return a serializable model configuration."""
        return asdict(self.config)
