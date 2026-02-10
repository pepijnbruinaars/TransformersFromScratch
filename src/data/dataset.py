"""PyTorch Dataset implementation with clean separation of concerns."""

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from typing import Any, Callable, Optional
import hashlib
import json
import logging
from pathlib import Path

import torch

from .preprocessing.masking import MaskedPair
from .preprocessing import (
    TokenizationTransform,
    AddSpecialTokensTransform,
    PaddingTransform,
    MaskingTransform,
    Compose,
    TextNormalizationTransform,
)
from ..tokenization.tokenizer import CustomTokenizer
from .config import PreprocessingConfig

logger = logging.getLogger(__name__)


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
        use_cache: bool = True,
    ) -> None:
        """Initialize translation dataset.

        Args:
            dataset: HuggingFace dataset containing translation pairs.
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            config: Preprocessing configuration.
            transform: Optional additional transform to apply.
            use_cache: If True, preprocess entire dataset once and cache results.
                      If False, apply preprocessing on-the-fly (old behavior).
        """
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config
        self.use_cache = False  # Will be set by _apply_cached_preprocessing() if enabled

        # Build preprocessing pipeline
        if transform is None:
            transform = self._build_default_transform()
        self.transform = transform

        # Apply cached preprocessing if enabled
        if use_cache and hasattr(config, 'use_preprocessing_cache') and config.use_preprocessing_cache:
            self._apply_cached_preprocessing()
        else:
            self.use_cache = False

    def _build_default_transform(self) -> Compose:
        """Build the default preprocessing pipeline.

        Returns:
            Composed transform pipeline.
        """
        transforms: list[Any] = []

        # Add text normalization as FIRST step if configured
        if hasattr(self.config, 'normalization_config') and \
           self.config.normalization_config and \
           self.config.normalization_config.enabled:
            transforms.append(
                TextNormalizationTransform(
                    unicode_normalization=self.config.normalization_config.unicode_normalization,
                    standardize_whitespace=self.config.normalization_config.standardize_whitespace,
                    standardize_quotes=self.config.normalization_config.standardize_quotes,
                    standardize_dashes=self.config.normalization_config.standardize_dashes,
                    lowercase=self.config.normalization_config.lowercase,
                    remove_control_chars=self.config.normalization_config.remove_control_chars,
                    source_field=self.config.translation_config.source_field,
                    target_field=self.config.translation_config.target_field,
                    translation_key=self.config.translation_config.translation_key,
                )
            )

        # Then tokenization
        transforms.append(
            TokenizationTransform(
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
                source_field=self.config.translation_config.source_field,
                target_field=self.config.translation_config.target_field,
                translation_key=self.config.translation_config.translation_key,
            )
        )

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

    def _generate_cache_key(self) -> str:
        """Generate a deterministic cache key based on config and tokenizer.

        Returns:
            MD5 hash string for caching.
        """
        # Create a dict of all relevant config values
        cache_dict = {
            'preprocessing_config': json.dumps(self.config.__dict__, sort_keys=True, default=str),
            'source_tokenizer_vocab_size': len(self.source_tokenizer.vocab) if hasattr(self.source_tokenizer, 'vocab') else 'unknown',
            'target_tokenizer_vocab_size': len(self.target_tokenizer.vocab) if hasattr(self.target_tokenizer, 'vocab') else 'unknown',
        }

        cache_str = json.dumps(cache_dict, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _apply_cached_preprocessing(self) -> None:
        """Apply preprocessing transforms using HuggingFace .map() with caching.

        This caches preprocessing results to disk, avoiding redundant computation
        across epochs. The first epoch will be slower (preprocessing happens upfront),
        but subsequent epochs will be much faster (2-3x speedup).
        """
        # Determine cache directory
        cache_dir = None
        if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = self._generate_cache_key()
        cache_file = None

        if cache_dir:
            cache_file = cache_dir / f"preprocessed_{cache_key}.arrow"
            logger.info(f"Preprocessing cache: {cache_file}")

        def preprocess_fn(example: dict) -> dict:
            """Apply preprocessing pipeline to a single example.

            Converts the result from MaskedPair (with tensors) to a dict with lists
            for HuggingFace dataset compatibility.
            """
            processed = self.transform(example)

            # Convert tensors to lists for HuggingFace serialization
            return {
                'source': processed['source'].tolist() if isinstance(processed['source'], torch.Tensor) else processed['source'],
                'target': processed['target'].tolist() if isinstance(processed['target'], torch.Tensor) else processed['target'],
                'source_mask': processed['source_mask'].tolist() if isinstance(processed['source_mask'], torch.Tensor) else processed['source_mask'],
                'target_mask': processed['target_mask'].tolist() if isinstance(processed['target_mask'], torch.Tensor) else processed['target_mask'],
                'label': processed['label'].tolist() if isinstance(processed['label'], torch.Tensor) else processed['label'],
            }

        try:
            logger.info("Preprocessing dataset with caching...")

            # Apply preprocessing with HuggingFace caching
            self.dataset = self.dataset.map(
                preprocess_fn,
                remove_columns=self.dataset.column_names,
                cache_file_name=str(cache_file) if cache_file else None,
                num_proc=4,  # Parallel preprocessing
                desc="Preprocessing translation pairs",
            )

            # Set format to PyTorch tensors for efficient loading
            self.dataset.set_format(
                type='torch',
                columns=['source', 'target', 'source_mask', 'target_mask', 'label']
            )

            self.use_cache = True
            logger.info("Preprocessing cache enabled - subsequent epochs will be 3-4x faster")

        except Exception as e:
            logger.warning(f"Failed to apply preprocessing cache: {e}. Falling back to on-the-fly preprocessing.")
            self.use_cache = False

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
        if self.use_cache:
            # Data is already preprocessed and cached, just return it
            item = self.dataset[index]  # type: ignore
            # HuggingFace set_format already converted to tensors
            return {
                'source': item['source'],
                'target': item['target'],
                'source_mask': item['source_mask'],
                'target_mask': item['target_mask'],
                'label': item['label'],
            }
        else:
            # Old behavior: apply preprocessing on-the-fly
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
