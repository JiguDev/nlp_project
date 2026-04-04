"""Evaluation metrics for language modeling and generation."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

try:
    import sacrebleu
except Exception:  # pragma: no cover - fallback when dependency is missing
    sacrebleu = None


def perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return float(math.exp(loss)) if loss < 50 else float("inf")


def compute_bleu(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute corpus BLEU for generated responses."""
    if sacrebleu is None:
        return 0.0
    bleu = sacrebleu.corpus_bleu(hypotheses, [list(references)])
    return float(bleu.score)


def exact_match_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Return the exact-match rate between two text collections."""
    if not references:
        return 0.0
    matches = sum(reference.strip() == hypothesis.strip() for reference, hypothesis in zip(references, hypotheses))
    return matches / len(references)
