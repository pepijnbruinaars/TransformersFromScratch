"""Multi-corpus data provider with weighted sampling."""

import logging
from typing import Optional
from datasets import Dataset, concatenate_datasets, interleave_datasets

from .base import DataProvider, DatasetSplit
from ..config import (
    DataProviderConfig,
    DatasetSplitConfig,
    MultiCorpusConfig,
    CategoryConfig,
    DatasetSpec,
)

logger = logging.getLogger(__name__)


class MultiCorpusDataProvider(DataProvider[Dataset]):
    """Data provider that combines multiple corpora with proportional sampling.

    This provider:
    1. Loads multiple datasets from different providers
    2. Calculates final proportions (category_proportion × dataset_proportion)
    3. Samples/truncates datasets to match target proportions
    4. Combines datasets with specified strategy (sequential or interleaved)
    5. Handles cases where datasets are smaller than target proportion

    The effective proportion for each dataset is:
        final_proportion = category_proportion × dataset_proportion
    """

    def __init__(
        self,
        config: MultiCorpusConfig,
        provider_registry: dict[str, type[DataProvider]],
    ) -> None:
        """Initialize multi-corpus data provider.

        Args:
            config: Multi-corpus configuration
            provider_registry: Mapping from provider names to provider classes
        """
        # Use a dummy DataProviderConfig for base class
        super().__init__(DataProviderConfig(
            name="multi_corpus",
            deterministic=True,
            seed=config.random_seed,
        ))
        self.multi_config = config
        self.provider_registry = provider_registry

    def _create_provider(
        self,
        provider_name: str,
        provider_config: DataProviderConfig
    ) -> DataProvider:
        """Create a provider instance from registry.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not found in registry
        """
        if provider_name not in self.provider_registry:
            raise ValueError(
                f"Provider '{provider_name}' not found in registry. "
                f"Available: {list(self.provider_registry.keys())}"
            )

        provider_class = self.provider_registry[provider_name]
        return provider_class(provider_config)

    def _calculate_final_proportions(self) -> list[tuple[CategoryConfig, DatasetSpec, float]]:
        """Calculate final proportions for all datasets.

        Returns:
            List of (category, dataset_spec, final_proportion) tuples
        """
        result = []
        for category in self.multi_config.categories:
            for dataset_spec in category.datasets:
                final_proportion = category.proportion * dataset_spec.proportion
                result.append((category, dataset_spec, final_proportion))
        return result

    def _load_and_resize_dataset(
        self,
        dataset_spec: DatasetSpec,
        target_size: int,
        category_name: str,
    ) -> Dataset:
        """Load a dataset and resize it to target size.

        Args:
            dataset_spec: Dataset specification
            target_size: Target number of examples
            category_name: Name of parent category (for logging)

        Returns:
            Resized dataset
        """
        # Create provider and load data
        provider = self._create_provider(
            dataset_spec.provider_name,
            dataset_spec.provider_config,
        )
        raw_dataset = provider.load_raw()
        dataset_size = len(raw_dataset)

        logger.info(
            f"  Dataset '{dataset_spec.provider_name}' has {dataset_size} examples "
            f"(target: {target_size})"
        )

        # Check if dataset is large enough
        if dataset_size < target_size:
            logger.warning(
                f"  WARNING: Dataset '{dataset_spec.provider_name}' in category '{category_name}' "
                f"is smaller than requested. Requested {target_size} examples, "
                f"but dataset only has {dataset_size} examples. Using all available data."
            )
            return raw_dataset

        # Sample/truncate to target size
        if dataset_size > target_size:
            # Shuffle and select
            shuffled = raw_dataset.shuffle(seed=self.config.seed)
            return shuffled.select(range(target_size))

        return raw_dataset

    def load_raw(self) -> Dataset:
        """Load and combine all datasets with proportional sampling.

        Returns:
            Combined dataset
        """
        logger.info(
            f"Loading multi-corpus dataset with {len(self.multi_config.categories)} "
            f"categories using '{self.multi_config.sampling_strategy}' strategy"
        )

        # Calculate final proportions
        dataset_info = self._calculate_final_proportions()

        # Log proportions
        logger.info("Category and dataset proportions:")
        for category, dataset_spec, final_prop in dataset_info:
            logger.info(
                f"  Category '{category.name}' ({category.proportion:.2f}) -> "
                f"Dataset '{dataset_spec.provider_name}' ({dataset_spec.proportion:.2f}) -> "
                f"Final proportion: {final_prop:.2f}"
            )

        # Determine target combined size
        # We need to load at least one dataset to estimate total size
        # For now, we'll load all datasets and determine size based on the largest one
        # that can fulfill its proportion
        logger.info("Loading datasets to determine combined size...")

        all_datasets: list[Dataset] = []
        all_proportions: list[float] = []
        actual_sizes: list[int] = []

        # First pass: load all datasets to determine sizes
        for category, dataset_spec, final_prop in dataset_info:
            provider = self._create_provider(
                dataset_spec.provider_name,
                dataset_spec.provider_config,
            )
            dataset = provider.load_raw()
            dataset_size = len(dataset)

            all_datasets.append(dataset)
            all_proportions.append(final_prop)
            actual_sizes.append(dataset_size)

            logger.info(
                f"  Loaded '{dataset_spec.provider_name}': {dataset_size} examples "
                f"(needs {final_prop:.2%} of combined)"
            )

        # Calculate target combined size
        # The combined size is limited by the smallest (dataset_size / proportion) ratio
        max_combined_size = min(
            actual_sizes[i] / all_proportions[i]
            for i in range(len(all_datasets))
        )
        max_combined_size = int(max_combined_size)

        logger.info(f"Target combined dataset size: {max_combined_size} examples")

        # Second pass: resize datasets to target proportions
        resized_datasets: list[Dataset] = []
        final_sizes: list[int] = []

        for i, (category, dataset_spec, final_prop) in enumerate(dataset_info):
            target_size = int(max_combined_size * final_prop)

            if len(all_datasets[i]) < target_size:
                logger.warning(
                    f"  WARNING: Dataset '{dataset_spec.provider_name}' "
                    f"is too small ({len(all_datasets[i])} < {target_size}). "
                    f"Using all {len(all_datasets[i])} examples."
                )
                resized_datasets.append(all_datasets[i])
                final_sizes.append(len(all_datasets[i]))
            else:
                # Sample to target size
                shuffled = all_datasets[i].shuffle(seed=self.config.seed)
                resized = shuffled.select(range(target_size))
                resized_datasets.append(resized)
                final_sizes.append(target_size)

            logger.info(
                f"  Using {final_sizes[-1]} examples from '{dataset_spec.provider_name}'"
            )

        # Combine datasets based on strategy
        if self.multi_config.sampling_strategy == "sequential":
            logger.info("Concatenating datasets sequentially")
            combined = concatenate_datasets(resized_datasets)

        elif self.multi_config.sampling_strategy == "interleaved":
            logger.info("Interleaving datasets with proportional sampling")

            # Since datasets are already sized proportionally, use equal probabilities
            # for interleaving
            probabilities = [size / sum(final_sizes) for size in final_sizes]

            logger.info(
                f"Interleave probabilities: "
                f"{[f'{p:.2%}' for p in probabilities]}"
            )

            combined = interleave_datasets(
                resized_datasets,
                probabilities=probabilities,
                seed=self.config.seed,
                stopping_strategy="all_exhausted",
            )
        else:
            raise ValueError(
                f"Unknown sampling strategy: {self.multi_config.sampling_strategy}"
            )

        logger.info(f"Combined dataset size: {len(combined)} examples")
        self.validate_dataset_schema(combined)

        return combined

    def get_all_sentences(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """Extract all source and target sentences from the dataset.

        Args:
            dataset: Dataset to extract sentences from

        Returns:
            Tuple of (source_sentences, target_sentences)
        """
        source_sentences: list[str] = []
        target_sentences: list[str] = []

        # Use translation config from preprocessing config
        translation_config = self.multi_config.preprocessing_config.translation_config

        for item in dataset:  # type: ignore
            translation = item[translation_config.translation_key]  # type: ignore
            source_sentences.append(translation[translation_config.source_field])
            target_sentences.append(translation[translation_config.target_field])

        return source_sentences, target_sentences

    def validate_dataset_schema(self, dataset: Dataset) -> None:
        """Validate that the dataset has the expected schema.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If schema is invalid
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Check first item
        first_item = dataset[0]
        translation_config = self.multi_config.preprocessing_config.translation_config

        if translation_config.translation_key not in first_item:
            raise ValueError(
                f"Expected key '{translation_config.translation_key}' not found in dataset"
            )

        translation = first_item[translation_config.translation_key]
        if not isinstance(translation, dict):
            raise ValueError(
                f"Expected '{translation_config.translation_key}' to be a dict"
            )

        required_fields = [
            translation_config.source_field,
            translation_config.target_field
        ]
        for field in required_fields:
            if field not in translation:
                raise ValueError(
                    f"Expected field '{field}' not found in translation dict"
                )
