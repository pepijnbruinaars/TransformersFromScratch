"""Data module for loading and preprocessing translation datasets.

This module provides a production-ready data pipeline with:
- Type-safe configuration
- Composable preprocessing transforms
- Multiple data providers (HuggingFace, local files)
- Efficient DataLoader factory
"""

from .config import (
    DatasetSplitConfig,
    SequenceConfig,
    TranslationConfig,
    PreprocessingConfig,
    DataLoaderConfig,
    DataProviderConfig,
    DataPipelineConfig,
)
from .providers.base import DataProvider, HuggingFaceDataProvider, LocalFileDataProvider
from .providers.euro_parl import EuroParlDataProvider
from .providers.opus_books import OpusBooksDataProvider
from .providers.wikimedia import WikimediaDataProvider
from .dataset import TranslationDataset, compute_max_sequence_length
from .dataloader import (
    DataLoaderFactory,
    create_dataloaders_from_config,
    compute_optimal_max_length,
)

__all__ = [
    # Configuration
    "DatasetSplitConfig",
    "SequenceConfig",
    "TranslationConfig",
    "PreprocessingConfig",
    "DataLoaderConfig",
    "DataProviderConfig",
    "DataPipelineConfig",
    # Providers
    "DataProvider",
    "HuggingFaceDataProvider",
    "LocalFileDataProvider",
    "EuroParlDataProvider",
    "OpusBooksDataProvider",
    "WikimediaDataProvider",
    # Dataset
    "TranslationDataset",
    "compute_max_sequence_length",
    # DataLoader
    "DataLoaderFactory",
    "create_dataloaders_from_config",
    "compute_optimal_max_length",
]
