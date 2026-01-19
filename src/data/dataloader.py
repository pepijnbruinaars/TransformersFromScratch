"""DataLoader factory functions with proper configuration."""

import logging
from typing import Optional
from torch.utils.data import DataLoader

from .dataset import TranslationDataset, compute_max_sequence_length
from .providers.base import DataProvider, DatasetSplit
from .config import (
    DataPipelineConfig,
    DataLoaderConfig,
    PreprocessingConfig,
)
from ..tokenization.tokenizer import CustomTokenizer


logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating DataLoaders with consistent configuration.

    This class encapsulates the entire data pipeline:
    1. Data provider loads and splits raw data
    2. Datasets apply preprocessing transforms
    3. DataLoaders handle batching and iteration
    """

    def __init__(
        self,
        provider: DataProvider,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        config: DataPipelineConfig,
    ) -> None:
        """Initialize DataLoader factory.

        Args:
            provider: Data provider for loading raw data.
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            config: Complete data pipeline configuration.
        """
        self.provider = provider
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config

        # Load and split data
        logger.info(f"Loading data using provider: {provider.__class__.__name__}")
        self.splits = provider.load(config.split_config)
        logger.info(
            f"Dataset splits - Train: {len(self.splits['train'])}, "  # type: ignore
            f"Val: {len(self.splits['val'])}, "  # type: ignore
            f"Test: {len(self.splits['test'])}"  # type: ignore
        )

    def _create_dataset(
        self,
        split_name: str,
        preprocessing_config: PreprocessingConfig,
    ) -> TranslationDataset:
        """Create a TranslationDataset for a specific split.

        Args:
            split_name: Name of the split ('train', 'val', or 'test').
            preprocessing_config: Preprocessing configuration.

        Returns:
            TranslationDataset instance.
        """
        dataset = TranslationDataset(
            dataset=self.splits[split_name],  # type: ignore
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            config=preprocessing_config,
        )
        return dataset

    def _create_dataloader(
        self,
        dataset: TranslationDataset,
        loader_config: DataLoaderConfig,
    ) -> DataLoader:
        """Create a DataLoader with the given configuration.

        Args:
            dataset: Dataset to load from.
            loader_config: DataLoader configuration.

        Returns:
            DataLoader instance.
        """
        dataloader_kwargs = {
            "batch_size": loader_config.batch_size,
            "shuffle": loader_config.shuffle,
            "num_workers": loader_config.num_workers,
            "pin_memory": loader_config.pin_memory,
            "drop_last": loader_config.drop_last,
        }

        # Add optional parameters if specified
        if loader_config.prefetch_factor is not None and loader_config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = loader_config.prefetch_factor

        if loader_config.persistent_workers and loader_config.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = loader_config.persistent_workers

        return DataLoader(dataset, **dataloader_kwargs)  # type: ignore

    def create_train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            Training DataLoader.
        """
        dataset = self._create_dataset("train", self.config.preprocessing_config)
        return self._create_dataloader(dataset, self.config.train_loader_config)

    def create_val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            Validation DataLoader.
        """
        dataset = self._create_dataset("val", self.config.preprocessing_config)
        return self._create_dataloader(dataset, self.config.val_loader_config)

    def create_test_dataloader(self) -> Optional[DataLoader]:
        """Create test DataLoader if configuration exists.

        Returns:
            Test DataLoader or None.
        """
        if self.config.test_loader_config is None:
            return None

        dataset = self._create_dataset("test", self.config.preprocessing_config)
        return self._create_dataloader(dataset, self.config.test_loader_config)

    def create_all_dataloaders(
        self,
    ) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create all DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_loader = self.create_train_dataloader()
        val_loader = self.create_val_dataloader()
        test_loader = self.create_test_dataloader()

        return train_loader, val_loader, test_loader


def create_dataloaders_from_config(
    provider: DataProvider,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    config: DataPipelineConfig,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Convenience function to create all DataLoaders from configuration.

    Args:
        provider: Data provider for loading raw data.
        source_tokenizer: Tokenizer for source language.
        target_tokenizer: Tokenizer for target language.
        config: Complete data pipeline configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    factory = DataLoaderFactory(
        provider=provider,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        config=config,
    )
    return factory.create_all_dataloaders()


def compute_optimal_max_length(
    provider: DataProvider,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    percentile: float = 0.95,
    max_cap: int = 512,
) -> int:
    """Compute optimal max_length based on dataset statistics.

    This function:
    1. Loads the full dataset
    2. Computes max sequence lengths
    3. Returns a reasonable max_length (capped and considering percentiles)

    Args:
        provider: Data provider for loading raw data.
        source_tokenizer: Tokenizer for source language.
        target_tokenizer: Tokenizer for target language.
        percentile: Percentile to use (e.g., 0.95 = 95th percentile).
        max_cap: Maximum allowed sequence length.

    Returns:
        Optimal max_length value.
    """
    # Load full dataset to compute statistics
    logger.info("Computing optimal sequence length from dataset...")
    raw_dataset = provider.load_raw()

    max_source, max_target = compute_max_sequence_length(
        dataset=raw_dataset,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    # Use the larger of source and target
    max_length = max(max_source, max_target)

    # Apply cap
    max_length = min(max_length, max_cap)

    logger.info(
        f"Computed max sequence length: {max_length} "
        f"(source: {max_source}, target: {max_target}, cap: {max_cap})"
    )

    return max_length
