"""OPUS Books dataset provider."""

from datasets import Dataset
from .base import HuggingFaceDataProvider
from ..config import DataProviderConfig


class OpusBooksDataProvider(HuggingFaceDataProvider):
    """Data provider for the opus_books dataset."""

    def __init__(self, config: DataProviderConfig) -> None:
        """Initialize OPUS Books data provider.

        Args:
            config: Data provider configuration.
        """
        super().__init__(
            config=config,
            dataset_name="opus_books",
            dataset_config="en-nl",
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
