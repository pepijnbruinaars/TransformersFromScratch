"""Tokenization transforms."""

from typing import TypedDict
import torch

from .base import Transform
from ...tokenization.tokenizer import CustomTokenizer
from ...constants import START_TOKEN, END_TOKEN


class RawTranslationPair(TypedDict):
    """Type for raw translation data."""
    translation: dict[str, str]


class TokenizedPair(TypedDict):
    """Type for tokenized translation pair."""
    source_ids: list[int]
    target_ids: list[int]
    source_text: str
    target_text: str


class TokenizationTransform(Transform[RawTranslationPair, TokenizedPair]):
    """Transform that tokenizes source and target sentences.

    This transform:
    1. Extracts source and target text from the translation pair
    2. Tokenizes both using the provided tokenizers
    3. Returns token IDs without special tokens (added later)
    """

    def __init__(
        self,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        source_field: str = "en",
        target_field: str = "nl",
        translation_key: str = "translation",
    ) -> None:
        """Initialize tokenization transform.

        Args:
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            source_field: Key for source language in translation dict.
            target_field: Key for target language in translation dict.
            translation_key: Key for translation dict in item.
        """
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_field = source_field
        self.target_field = target_field
        self.translation_key = translation_key

    def __call__(self, item: RawTranslationPair) -> TokenizedPair:
        """Tokenize a translation pair.

        Args:
            item: Raw translation pair.

        Returns:
            Tokenized pair with token IDs and original text.
        """
        translation = item[self.translation_key]
        source_text: str = translation[self.source_field]
        target_text: str = translation[self.target_field]

        # Tokenize (returns list of token IDs)
        source_ids = self.source_tokenizer.encode(source_text)
        target_ids = self.target_tokenizer.encode(target_text)

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_text": source_text,
            "target_text": target_text,
        }


class TensorizedPair(TypedDict):
    """Type for tensorized translation pair with special tokens."""
    source_tensor: torch.Tensor
    target_tensor: torch.Tensor
    label_tensor: torch.Tensor
    source_text: str
    target_text: str


class AddSpecialTokensTransform(Transform[TokenizedPair, TensorizedPair]):
    """Transform that adds special tokens and converts to tensors.

    For sequence-to-sequence tasks:
    - Source: [START] + tokens + [END]
    - Target (decoder input): [START] + tokens
    - Label (decoder output): tokens + [END]
    """

    def __init__(
        self,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
    ) -> None:
        """Initialize special tokens transform.

        Args:
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
        """
        self.source_start_id = source_tokenizer.token_to_id(START_TOKEN)
        self.source_end_id = source_tokenizer.token_to_id(END_TOKEN)
        self.target_start_id = target_tokenizer.token_to_id(START_TOKEN)
        self.target_end_id = target_tokenizer.token_to_id(END_TOKEN)

    def __call__(self, item: TokenizedPair) -> TensorizedPair:
        """Add special tokens and convert to tensors.

        Args:
            item: Tokenized pair.

        Returns:
            Tensorized pair with special tokens.
        """
        source_ids = item["source_ids"]
        target_ids = item["target_ids"]

        # Source: [START] + tokens + [END]
        source_tensor = torch.tensor(
            [self.source_start_id] + source_ids + [self.source_end_id],
            dtype=torch.int64,
        )

        # Target (decoder input): [START] + tokens
        target_tensor = torch.tensor(
            [self.target_start_id] + target_ids,
            dtype=torch.int64,
        )

        # Label (expected output): tokens + [END]
        label_tensor = torch.tensor(
            target_ids + [self.target_end_id],
            dtype=torch.int64,
        )

        return {
            "source_tensor": source_tensor,
            "target_tensor": target_tensor,
            "label_tensor": label_tensor,
            "source_text": item["source_text"],
            "target_text": item["target_text"],
        }
