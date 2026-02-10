"""OpenSubtitles data provider for translation tasks."""

import logging
from datasets import Dataset, Features, load_dataset  # type: ignore
from datasets.features import Translation  # type: ignore

from .base import HuggingFaceDataProvider
from ..config import DataProviderConfig

logger = logging.getLogger(__name__)

# Mapping from language codes to the dataset's naming convention
# The sentence-transformers dataset uses "en-XX" format for subsets
SUPPORTED_LANGUAGE_PAIRS = {
    ("en", "nl"): "en-nl",
    ("en", "de"): "en-de",
    ("en", "fr"): "en-fr",
    ("en", "es"): "en-es",
    ("en", "it"): "en-it",
    ("en", "pt"): "en-pt",
    ("en", "ru"): "en-ru",
    ("en", "zh"): "en-zh",
    ("en", "ja"): "en-ja",
    ("en", "ko"): "en-ko",
    ("en", "ar"): "en-ar",
    ("en", "tr"): "en-tr",
    ("en", "pl"): "en-pl",
    ("en", "sv"): "en-sv",
    ("en", "da"): "en-da",
    ("en", "fi"): "en-fi",
    ("en", "no"): "en-no",
    ("en", "cs"): "en-cs",
    ("en", "hu"): "en-hu",
    ("en", "ro"): "en-ro",
}


class OpenSubtitlesDataProvider(HuggingFaceDataProvider):
    """Data provider for OpenSubtitles dataset.

    Uses the Parquet-backed sentence-transformers/parallel-sentences-opensubtitles
    dataset which works with modern datasets library (>=4.0.0) without requiring
    trust_remote_code or script-based loading.

    The dataset contains parallel corpora extracted from movie and TV subtitles
    in multiple languages.

    Example usage:
        config = DataProviderConfig(
            name="open_subtitles",
            dataset_name="open_subtitles",
            lang1="en",
            lang2="nl",
            split="train",
        )
        provider = OpenSubtitlesDataProvider(config)
    """

    def __init__(self, config: DataProviderConfig) -> None:
        """Initialize OpenSubtitles data provider.

        Args:
            config: Data provider configuration.
        """
        super().__init__(
            config=config,
            dataset_name="sentence-transformers/parallel-sentences-opensubtitles",
            dataset_config=None,
        )

    def load_raw(self) -> Dataset:
        """Load the OpenSubtitles dataset from HuggingFace.

        Returns:
            Dataset: Raw OpenSubtitles dataset with standardized 'translation' column

        Raises:
            ValueError: If lang1 or lang2 not specified in config, or unsupported pair
        """
        if not hasattr(self.config, "lang1") or not hasattr(self.config, "lang2"):
            raise ValueError(
                "OpenSubtitles provider requires 'lang1' and 'lang2' parameters. "
                "Example: lang1='en', lang2='nl'"
            )

        lang1 = self.config.lang1
        lang2 = self.config.lang2

        # Determine the subset name - dataset uses "en-XX" format
        lang_pair = (lang1, lang2)
        lang_pair_reversed = (lang2, lang1)

        if lang_pair in SUPPORTED_LANGUAGE_PAIRS:
            subset = SUPPORTED_LANGUAGE_PAIRS[lang_pair]  # type: ignore
            source_col, target_col = "english", "non_english"
        elif lang_pair_reversed in SUPPORTED_LANGUAGE_PAIRS:
            subset = SUPPORTED_LANGUAGE_PAIRS[lang_pair_reversed]  # type: ignore
            source_col, target_col = "non_english", "english"
        else:
            raise ValueError(
                f"Unsupported language pair: {lang1}-{lang2}. "
                f"Supported pairs (with English): {list(SUPPORTED_LANGUAGE_PAIRS.keys())}"
            )

        logger.info(f"Loading OpenSubtitles dataset: {subset} (Parquet-backed)")

        dataset = load_dataset(
            "sentence-transformers/parallel-sentences-opensubtitles",
            subset,
            split=self.config.split,
            cache_dir=self.config.cache_dir,
        )

        logger.info(f"Loaded {len(dataset)} examples from OpenSubtitles")

        # Transform to standardized translation format
        def to_translation(example: dict) -> dict:
            return {
                "translation": {
                    lang1: example[source_col],
                    lang2: example[target_col],
                }
            }

        # Define features with proper Translation type for compatibility with other corpora
        target_features = Features(
            {"translation": Translation(languages=[lang1, lang2])}
        )

        dataset = dataset.map(
            to_translation,
            remove_columns=dataset.column_names,
            features=target_features,
            desc="Converting to translation format",
        )

        self.validate_dataset_schema(dataset)
        return dataset

    def get_all_sentences(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """Extract all source and target sentences from the dataset.

        Args:
            dataset: Dataset to extract sentences from.

        Returns:
            Tuple of (source_sentences, target_sentences) using lang1 and lang2.
        """
        source_sentences: list[str] = []
        target_sentences: list[str] = []

        for item in dataset:  # type: ignore
            translation = item["translation"]  # type: ignore
            source_sentences.append(translation[self.config.lang1])
            target_sentences.append(translation[self.config.lang2])

        return source_sentences, target_sentences
