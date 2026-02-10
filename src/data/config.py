"""Data pipeline configuration classes with full type safety."""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Configuration for dataset splitting ratios.

    All split ratios must sum to 1.0.
    """
    train: float
    val: float
    test: float
    max_train_size: Optional[int] = None  # Cap on training set size

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Train, validation, and test splits must sum to 1.0, got {total}"
            )
        if any(split < 0 for split in (self.train, self.val, self.test)):
            raise ValueError("All split ratios must be non-negative")
        if self.max_train_size is not None and self.max_train_size <= 0:
            raise ValueError(f"max_train_size must be positive, got {self.max_train_size}")


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for sequence processing."""
    max_length: int
    truncation: bool = True
    padding: Literal["max_length", "do_not_pad"] = "max_length"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")

@dataclass(frozen=True)
class TranslationConfig:
    """Configuration for translation task."""
    source_lang: str
    target_lang: str
    source_field: str = "en"
    target_field: str = "nl"
    translation_key: str = "translation"

    def __post_init__(self) -> None:
        if not self.source_lang or not self.target_lang:
            raise ValueError("Source and target languages must be specified")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    sequence_config: SequenceConfig
    translation_config: TranslationConfig
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_causal_mask: bool = True
    cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            object.__setattr__(self, 'cache_dir', Path(self.cache_dir))


@dataclass
class DataLoaderConfig:
    """Configuration for PyTorch DataLoader."""
    batch_size: int
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")


@dataclass(frozen=True)
class DataProviderConfig:
    """Configuration for data provider."""
    name: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "train"
    cache_dir: Optional[Path] = None
    local_files: Optional[dict[str, Path]] = None
    deterministic: bool = True
    seed: int = 42
    # Language pair parameters (for OpenSubtitles and similar datasets)
    lang1: Optional[str] = None
    lang2: Optional[str] = None

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            object.__setattr__(self, 'cache_dir', Path(self.cache_dir))
        if self.local_files is not None:
            object.__setattr__(
                self,
                'local_files',
                {k: Path(v) for k, v in self.local_files.items()}
            )


@dataclass(frozen=True)
class DataPipelineConfig:
    """Complete data pipeline configuration."""
    provider_config: DataProviderConfig
    split_config: DatasetSplitConfig
    preprocessing_config: PreprocessingConfig
    train_loader_config: DataLoaderConfig
    val_loader_config: DataLoaderConfig
    test_loader_config: Optional[DataLoaderConfig] = None

    @classmethod
    def create_default(
        cls,
        provider_name: str,
        max_sequence_length: int = 512,
        batch_size: int = 8,
    ) -> "DataPipelineConfig":
        """Create a default configuration for common use cases."""
        return cls(
            provider_config=DataProviderConfig(
                name=provider_name,
                deterministic=True,
                seed=42,
            ),
            split_config=DatasetSplitConfig(
                train=0.7,
                val=0.15,
                test=0.15,
            ),
            preprocessing_config=PreprocessingConfig(
                sequence_config=SequenceConfig(
                    max_length=max_sequence_length,
                    truncation=True,
                    padding="max_length",
                ),
                translation_config=TranslationConfig(
                    source_lang="en",
                    target_lang="nl",
                ),
            ),
            train_loader_config=DataLoaderConfig(
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            ),
            val_loader_config=DataLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            ),
        )


@dataclass(frozen=True)
class TextNormalizationConfig:
    """Configuration for text normalization preprocessing.

    Applied before tokenization to standardize text.
    """
    enabled: bool = True
    unicode_normalization: str = "NFKC"  # "NFC", "NFKC", "NFD", "NFKD", "none"
    standardize_whitespace: bool = True
    standardize_quotes: bool = True  # ' → ', " → "
    standardize_dashes: bool = True  # – → -, — → -
    lowercase: bool = False
    remove_control_chars: bool = True

    def __post_init__(self) -> None:
        valid_norms = {"NFC", "NFKC", "NFD", "NFKD", "none"}
        if self.unicode_normalization not in valid_norms:
            raise ValueError(
                f"unicode_normalization must be one of {valid_norms}, "
                f"got {self.unicode_normalization}"
            )


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a single dataset within a corpus category.

    Defines which dataset to use and what proportion of the category it represents.
    """
    provider_name: str  # e.g., "europarl", "opus_books", "wikimedia"
    provider_config: DataProviderConfig  # Full config for this provider
    proportion: float = 1.0  # Proportion within category (0.0-1.0)

    def __post_init__(self) -> None:
        if not (0.0 < self.proportion <= 1.0):
            raise ValueError(
                f"proportion must be between 0 and 1, got {self.proportion}"
            )


@dataclass(frozen=True)
class CategoryConfig:
    """Configuration for a category of datasets.

    A category groups related datasets (e.g., "legal", "literary", "general")
    with a category-level proportion and individual dataset proportions.
    Dataset proportions within the category must sum to 1.0.
    """
    name: str
    proportion: float  # Proportion of final combined dataset (0.0-1.0)
    datasets: tuple[DatasetSpec, ...]  # Multiple datasets in this category

    def __post_init__(self) -> None:
        if not (0.0 < self.proportion <= 1.0):
            raise ValueError(
                f"Category proportion must be between 0 and 1, got {self.proportion}"
            )
        if len(self.datasets) == 0:
            raise ValueError("Category must have at least one dataset")

        # Validate dataset proportions sum to 1.0
        dataset_proportion_sum = sum(ds.proportion for ds in self.datasets)
        if not (0.99 <= dataset_proportion_sum <= 1.01):
            raise ValueError(
                f"Dataset proportions within category '{self.name}' must sum to 1.0, "
                f"got {dataset_proportion_sum}"
            )


@dataclass(frozen=True)
class MultiCorpusConfig:
    """Configuration for multi-corpus training with weighted sampling.

    This config allows:
    1. Grouping datasets into categories (e.g., legal, literary, general)
    2. Setting proportions at both category and dataset levels
    3. Text normalization before tokenization
    4. Two sampling strategies: sequential or interleaved

    Category proportions must sum to 1.0.
    Dataset proportions within each category must sum to 1.0.
    """
    categories: tuple[CategoryConfig, ...]
    split_config: DatasetSplitConfig
    preprocessing_config: PreprocessingConfig
    train_loader_config: DataLoaderConfig
    val_loader_config: DataLoaderConfig
    test_loader_config: Optional[DataLoaderConfig] = None
    normalization_config: Optional[TextNormalizationConfig] = None
    sampling_strategy: Literal["sequential", "interleaved"] = "interleaved"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if len(self.categories) == 0:
            raise ValueError("At least one category is required")

        # Validate category proportions sum to 1.0
        category_proportion_sum = sum(cat.proportion for cat in self.categories)
        if not (0.99 <= category_proportion_sum <= 1.01):
            raise ValueError(
                f"Category proportions must sum to 1.0, got {category_proportion_sum}"
            )

        # Set default normalization config if not provided
        if self.normalization_config is None:
            object.__setattr__(
                self,
                'normalization_config',
                TextNormalizationConfig()
            )
