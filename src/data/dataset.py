"""PyTorch Dataset implementation with clean separation of concerns."""

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from typing import Any, Callable, Optional

from .preprocessing.masking import MaskedPair
from .preprocessing import (
    TokenizationTransform,
    AddSpecialTokensTransform,
    PaddingTransform,
    MaskingTransform,
    Compose,
)
from ..tokenization.tokenizer import CustomTokenizer
from .config import PreprocessingConfig


class TranslationDataset(TorchDataset):
    """PyTorch Dataset for translation tasks with composable preprocessing.

    This dataset:
    1. Wraps a HuggingFace Dataset
    2. Applies preprocessing transforms on-the-fly in __getitem__
    3. Keeps preprocessing logic separate from data loading

    The preprocessing pipeline is:
    Raw data -> Tokenization -> Special tokens -> Padding -> Masking -> Output
    """

    def __init__(
        self,
        dataset: Dataset,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        config: PreprocessingConfig,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize translation dataset.

        Args:
            dataset: HuggingFace dataset containing translation pairs.
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            config: Preprocessing configuration.
            transform: Optional additional transform to apply.
        """
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config

        # Build preprocessing pipeline
        if transform is None:
            transform = self._build_default_transform()
        self.transform = transform

    def _build_default_transform(self) -> Compose:
        """Build the default preprocessing pipeline.

        Returns:
            Composed transform pipeline.
        """
        transforms: list[Any] = [
            TokenizationTransform(
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
                source_field=self.config.translation_config.source_field,
                target_field=self.config.translation_config.target_field,
                translation_key=self.config.translation_config.translation_key,
            ),
        ]

        if self.config.add_special_tokens:
            transforms.append(
                AddSpecialTokensTransform(
                    source_tokenizer=self.source_tokenizer,
                    target_tokenizer=self.target_tokenizer,
                )
            )

        transforms.append(
            PaddingTransform(
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
                max_length=self.config.sequence_config.max_length,
                padding=self.config.sequence_config.padding,
                truncation=self.config.sequence_config.truncation,
            )
        )

        if self.config.return_attention_mask:
            transforms.append(
                MaskingTransform(
                    source_tokenizer=self.source_tokenizer,
                    target_tokenizer=self.target_tokenizer,
                    sequence_length=self.config.sequence_config.max_length,
                )
            )

        return Compose(transforms)  # type: ignore

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> MaskedPair:
        """Get a preprocessed item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            Preprocessed translation pair with masks.
        """
        # Get raw item from HuggingFace dataset
        raw_item = self.dataset[index]  # type: ignore

        # Apply preprocessing pipeline
        processed_item: MaskedPair = self.transform(raw_item)  # type: ignore

        return processed_item


def compute_max_sequence_length(
    dataset: Dataset,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    source_field: str = "en",
    target_field: str = "nl",
    translation_key: str = "translation",
) -> tuple[int, int]:
    """Compute maximum sequence lengths in a dataset.

    This is useful for determining appropriate max_length values.

    Args:
        dataset: Dataset to analyze.
        source_tokenizer: Tokenizer for source language.
        target_tokenizer: Tokenizer for target language.
        source_field: Key for source language in translation dict.
        target_field: Key for target language in translation dict.
        translation_key: Key for translation dict in item.

    Returns:
        Tuple of (max_source_length, max_target_length).
    """
    max_source_length = 0
    max_target_length = 0

    for item in dataset:  # type: ignore
        translation = item[translation_key]  # type: ignore
        source_text = translation[source_field]
        target_text = translation[target_field]

        source_length = len(source_tokenizer.encode(source_text))
        target_length = len(target_tokenizer.encode(target_text))

        max_source_length = max(max_source_length, source_length)
        max_target_length = max(max_target_length, target_length)

    # Add space for special tokens (START and END)
    return max_source_length + 2, max_target_length + 2
