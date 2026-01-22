"""Base classes for data providers with proper type safety."""

from abc import ABC, abstractmethod
from typing import TypedDict, Generic, TypeVar
from datasets import Dataset

from ..config import DataProviderConfig, DatasetSplitConfig


class TranslationPair(TypedDict):
    """Type for a single translation pair."""
    translation: dict[str, str]


class DatasetSplit(TypedDict):
    """Type for dataset splits."""
    train: Dataset
    val: Dataset
    test: Dataset


T = TypeVar('T')


class DataProvider(ABC, Generic[T]):
    """Abstract base class for data providers.

    A DataProvider is responsible for:
    1. Loading raw data from a source (HuggingFace, local files, etc.)
    2. Splitting data into train/val/test sets
    3. Returning data in a standardized format

    Type parameter T represents the raw data type before processing.
    """

    def __init__(self, config: DataProviderConfig) -> None:
        """Initialize the data provider.

        Args:
            config: Configuration for the data provider.
        """
        self.config = config

    @abstractmethod
    def load_raw(self) -> Dataset:
        """Load the raw dataset from the source.

        Returns:
            Raw dataset without any splits.

        Raises:
            FileNotFoundError: If local files are required but not found.
            ValueError: If dataset configuration is invalid.
        """
        pass

    def split_dataset(
        self,
        dataset: Dataset,
        split_config: DatasetSplitConfig,
    ) -> DatasetSplit:
        """Split dataset into train/val/test sets.

        Args:
            dataset: The full dataset to split.
            split_config: Configuration specifying split ratios.

        Returns:
            Dictionary containing train, val, and test datasets.
        """
        # Shuffle dataset if deterministic mode is enabled
        if self.config.deterministic:
            dataset = dataset.shuffle(seed=self.config.seed)

        total_size = len(dataset)
        
        # Calculate initial train size
        train_size = int(total_size * split_config.train)
        
        # Apply cap if specified
        if split_config.max_train_size is not None:
            train_size = min(train_size, split_config.max_train_size)
        
        # Recalculate val and test sizes based on remaining data
        remaining_size = total_size - train_size
        val_size = int(remaining_size * (split_config.val / (split_config.val + split_config.test)))
        
        # Create splits using select for efficiency
        train_dataset = dataset.select(range(0, train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def load(self, split_config: DatasetSplitConfig) -> DatasetSplit:
        """Load and split the dataset.

        This is the main public interface for data providers.

        Args:
            split_config: Configuration specifying split ratios.

        Returns:
            Dictionary containing train, val, and test datasets.
        """
        raw_dataset = self.load_raw()
        return self.split_dataset(raw_dataset, split_config)

    @abstractmethod
    def get_all_sentences(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """Extract all source and target sentences from the dataset.

        Useful for training tokenizers or computing statistics.

        Args:
            dataset: Dataset to extract sentences from.

        Returns:
            Tuple of (source_sentences, target_sentences).
        """
        pass

    def validate_dataset_schema(self, dataset: Dataset) -> None:
        """Validate that the dataset has the expected schema.

        Args:
            dataset: Dataset to validate.

        Raises:
            ValueError: If dataset schema is invalid.
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Check first item has expected structure
        first_item = dataset[0]
        if not isinstance(first_item, dict):
            raise ValueError("Dataset items must be dictionaries")


class HuggingFaceDataProvider(DataProvider[Dataset]):
    """Base class for providers that load from HuggingFace datasets."""

    def __init__(
        self,
        config: DataProviderConfig,
        dataset_name: str,
        dataset_config: str | None = None,
    ) -> None:
        """Initialize HuggingFace data provider.

        Args:
            config: Data provider configuration.
            dataset_name: Name of the HuggingFace dataset.
            dataset_config: Optional dataset configuration (e.g., language pair).
        """
        super().__init__(config)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def load_raw(self) -> Dataset:
        """Load dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.config.split,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
        )

        if not isinstance(dataset, Dataset):
            raise ValueError(
                f"Expected Dataset, got {type(dataset)}. "
                "Ensure split parameter returns a single dataset."
            )

        self.validate_dataset_schema(dataset)
        return dataset


class LocalFileDataProvider(DataProvider[Dataset]):
    """Base class for providers that load from local files."""

    def __init__(
        self,
        config: DataProviderConfig,
        source_file: str,
        target_file: str,
    ) -> None:
        """Initialize local file data provider.

        Args:
            config: Data provider configuration.
            source_file: Path to source language file.
            target_file: Path to target language file.
        """
        super().__init__(config)
        self.source_file = source_file
        self.target_file = target_file

    def load_raw(self) -> Dataset:
        """Load dataset from local files."""
        from pathlib import Path
        from datasets import Dataset as HFDataset

        source_path = Path(self.source_file)
        target_path = Path(self.target_file)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        # Read sentences
        with open(source_path, 'r', encoding='utf-8') as f:
            source_sentences = [line.strip() for line in f if line.strip()]

        with open(target_path, 'r', encoding='utf-8') as f:
            target_sentences = [line.strip() for line in f if line.strip()]

        if len(source_sentences) != len(target_sentences):
            raise ValueError(
                f"Mismatch in number of sentences: "
                f"source={len(source_sentences)}, target={len(target_sentences)}"
            )

        # Create HuggingFace Dataset
        dataset = HFDataset.from_dict({
            "translation": [
                {"en": src, "nl": tgt}
                for src, tgt in zip(source_sentences, target_sentences)
            ]
        })

        self.validate_dataset_schema(dataset)
        return dataset
