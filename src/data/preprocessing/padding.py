"""Padding and truncation transforms."""

from typing import TypedDict, Literal
import torch

from .base import Transform
from .tokenization import TensorizedPair
from ...tokenization.tokenizer import CustomTokenizer
from ...constants import PAD_TOKEN


class PaddedPair(TypedDict):
    """Type for padded translation pair."""
    source: torch.Tensor
    target: torch.Tensor
    label: torch.Tensor
    source_text: str
    target_text: str


class PaddingTransform(Transform[TensorizedPair, PaddedPair]):
    """Transform that handles padding and truncation.

    This transform:
    1. Truncates sequences that exceed max_length
    2. Pads sequences to max_length or batch max length
    """

    def __init__(
        self,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        max_length: int,
        padding: Literal["max_length", "do_not_pad"] = "max_length",
        truncation: bool = True,
    ) -> None:
        """Initialize padding transform.

        Args:
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            max_length: Maximum sequence length.
            padding: Padding strategy ("max_length" or "do_not_pad").
            truncation: Whether to truncate sequences exceeding max_length.
        """
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.source_pad_id = source_tokenizer.token_to_id(PAD_TOKEN)
        self.target_pad_id = target_tokenizer.token_to_id(PAD_TOKEN)

    def _truncate_and_pad(
        self,
        tensor: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        """Truncate and pad a tensor.

        Args:
            tensor: Input tensor.
            pad_id: Padding token ID.

        Returns:
            Truncated and padded tensor.
        """
        # Truncate if necessary
        if self.truncation and len(tensor) > self.max_length:
            tensor = tensor[: self.max_length]

        # Pad if necessary
        if self.padding == "max_length":
            num_padding = self.max_length - len(tensor)
            if num_padding > 0:
                padding_tensor = torch.full(
                    (num_padding,),
                    pad_id,
                    dtype=torch.int64,
                )
                tensor = torch.cat([tensor, padding_tensor])

        return tensor

    def __call__(self, item: TensorizedPair) -> PaddedPair:
        """Apply padding and truncation.

        Args:
            item: Tensorized pair.

        Returns:
            Padded pair.
        """
        source_tensor = self._truncate_and_pad(
            item["source_tensor"],
            self.source_pad_id,
        )
        target_tensor = self._truncate_and_pad(
            item["target_tensor"],
            self.target_pad_id,
        )
        label_tensor = self._truncate_and_pad(
            item["label_tensor"],
            self.target_pad_id,
        )

        return {
            "source": source_tensor,
            "target": target_tensor,
            "label": label_tensor,
            "source_text": item["source_text"],
            "target_text": item["target_text"],
        }
