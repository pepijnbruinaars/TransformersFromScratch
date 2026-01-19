"""Masking transforms for attention."""

from typing import TypedDict
import torch

from .base import Transform
from .padding import PaddedPair
from ...tokenization.tokenizer import CustomTokenizer
from ...constants import PAD_TOKEN


class MaskedPair(TypedDict):
    """Type for masked translation pair (final output)."""
    source: torch.Tensor
    target: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    label: torch.Tensor
    source_text: str
    target_text: str


def create_padding_mask(
    tensor: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Create a padding mask for a sequence.

    Args:
        tensor: Input sequence tensor.
        pad_id: Padding token ID.

    Returns:
        Boolean mask of shape (1, 1, seq_len) where True means keep.
    """
    return (tensor != pad_id).unsqueeze(0).unsqueeze(0)


def create_causal_mask(size: int) -> torch.Tensor:
    """Create a causal (upper triangular) mask.

    Args:
        size: Sequence length.

    Returns:
        Boolean mask of shape (1, size, size) where True means keep.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class MaskingTransform(Transform[PaddedPair, MaskedPair]):
    """Transform that creates attention masks.

    This transform creates:
    1. Source mask: padding mask only (encoder can see all positions)
    2. Target mask: padding mask AND causal mask (decoder is autoregressive)
    """

    def __init__(
        self,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        sequence_length: int,
    ) -> None:
        """Initialize masking transform.

        Args:
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            sequence_length: Maximum sequence length for pre-computing causal mask.
        """
        self.source_pad_id = source_tokenizer.token_to_id(PAD_TOKEN)
        self.target_pad_id = target_tokenizer.token_to_id(PAD_TOKEN)

        # Pre-compute causal mask for efficiency
        self._causal_mask = create_causal_mask(sequence_length)

    def __call__(self, item: PaddedPair) -> MaskedPair:
        """Create attention masks.

        Args:
            item: Padded pair.

        Returns:
            Masked pair with attention masks.
        """
        source = item["source"]
        target = item["target"]

        # Source mask: only padding mask
        source_mask = create_padding_mask(source, self.source_pad_id)

        # Target mask: padding mask AND causal mask
        target_padding_mask = create_padding_mask(target, self.target_pad_id)
        target_mask = target_padding_mask & self._causal_mask

        return {
            "source": source,
            "target": target,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "label": item["label"],
            "source_text": item["source_text"],
            "target_text": item["target_text"],
        }
