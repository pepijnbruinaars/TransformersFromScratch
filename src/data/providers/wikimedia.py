"""Wikimedia dataset provider (from local files)."""

from datasets import Dataset
from .base import LocalFileDataProvider
from ..config import DataProviderConfig


class WikimediaDataProvider(LocalFileDataProvider):
    """Data provider for Wikimedia dataset stored in local files."""

    def __init__(self, config: DataProviderConfig) -> None:
        """Initialize Wikimedia data provider.

        Args:
            config: Data provider configuration.
        """
        super().__init__(
            config=config,
            source_file="data/processed/wikimedia/wikimedia.en-nl.en.cleaned",
            target_file="data/processed/wikimedia/wikimedia.en-nl.nl.cleaned",
        )

    def get_all_sentences(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """Extract all source and target sentences from the dataset.

        Args:
            dataset: Dataset to extract sentences from.

        Returns:
            Tuple of (source_sentences, target_sentences).
        """
        source_sentences: list[str] = []
        target_sentences: list[str] = []

        for item in dataset:  # type: ignore
            translation = item["translation"]  # type: ignore
            source_sentences.append(translation["en"])
            target_sentences.append(translation["nl"])

        return source_sentences, target_sentences
