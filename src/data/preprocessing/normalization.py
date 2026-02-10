"""Text normalization transforms."""

import unicodedata
import re
from typing import TypedDict

from .base import Transform


class RawTranslationPair(TypedDict):
    """Type for raw translation data (same as in tokenization.py)."""
    translation: dict[str, str]


class TextNormalizationTransform(Transform[RawTranslationPair, RawTranslationPair]):
    """Transform that normalizes text before tokenization.

    This transform applies various text normalization operations:
    1. Unicode normalization (NFKC by default)
    2. Whitespace standardization
    3. Quote standardization (' → ', " → ")
    4. Dash standardization (– → -, — → -)
    5. Control character removal
    6. Optional lowercasing

    Applied BEFORE tokenization in the preprocessing pipeline.
    """

    # Quote mappings
    QUOTE_MAPPINGS = {
        '\u2018': "'",  # ' → '
        '\u2019': "'",  # ' → '
        '\u201a': "'",  # ‚ → '
        '\u201b': "'",  # ‛ → '
        '\u201c': '"',  # " → "
        '\u201d': '"',  # " → "
        '\u201e': '"',  # „ → "
        '\u201f': '"',  # ‟ → "
        '\u2039': "'",  # ‹ → '
        '\u203a': "'",  # › → '
    }

    # Dash mappings
    DASH_MAPPINGS = {
        '\u2013': '-',  # – (en dash) → -
        '\u2014': '-',  # — (em dash) → -
        '\u2015': '-',  # ― (horizontal bar) → -
        '\u2212': '-',  # − (minus sign) → -
    }

    def __init__(
        self,
        unicode_normalization: str = "NFKC",
        standardize_whitespace: bool = True,
        standardize_quotes: bool = True,
        standardize_dashes: bool = True,
        lowercase: bool = False,
        remove_control_chars: bool = True,
        source_field: str = "en",
        target_field: str = "nl",
        translation_key: str = "translation",
    ) -> None:
        """Initialize text normalization transform.

        Args:
            unicode_normalization: Unicode normalization form ("NFC", "NFKC", "NFD", "NFKD", "none")
            standardize_whitespace: Replace multiple spaces/tabs with single space
            standardize_quotes: Standardize various quote characters
            standardize_dashes: Standardize various dash characters
            lowercase: Convert to lowercase
            remove_control_chars: Remove control characters
            source_field: Key for source language in translation dict
            target_field: Key for target language in translation dict
            translation_key: Key for translation dict in item
        """
        self.unicode_normalization = unicode_normalization
        self.standardize_whitespace = standardize_whitespace
        self.standardize_quotes = standardize_quotes
        self.standardize_dashes = standardize_dashes
        self.lowercase = lowercase
        self.remove_control_chars = remove_control_chars
        self.source_field = source_field
        self.target_field = target_field
        self.translation_key = translation_key

        # Pre-compile regex patterns for efficiency
        if self.standardize_whitespace:
            self._whitespace_pattern = re.compile(r'\s+')
        if self.remove_control_chars:
            # Keep newlines, tabs, and normal spaces
            self._control_chars_pattern = re.compile(
                r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]'
            )

    def normalize_text(self, text: str) -> str:
        """Apply all normalization steps to a text string.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # 1. Remove control characters
        if self.remove_control_chars:
            text = self._control_chars_pattern.sub('', text)

        # 2. Unicode normalization
        if self.unicode_normalization != "none":
            text = unicodedata.normalize(self.unicode_normalization, text)

        # 3. Standardize quotes
        if self.standardize_quotes:
            for old, new in self.QUOTE_MAPPINGS.items():
                text = text.replace(old, new)

        # 4. Standardize dashes
        if self.standardize_dashes:
            for old, new in self.DASH_MAPPINGS.items():
                text = text.replace(old, new)

        # 5. Standardize whitespace
        if self.standardize_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
            text = text.strip()

        # 6. Lowercase (optional)
        if self.lowercase:
            text = text.lower()

        return text

    def __call__(self, item: RawTranslationPair) -> RawTranslationPair:
        """Apply text normalization to a translation pair.

        Args:
            item: Raw translation pair

        Returns:
            Translation pair with normalized text
        """
        translation = item[self.translation_key]
        source_text = translation[self.source_field]
        target_text = translation[self.target_field]

        # Normalize both source and target
        normalized_source = self.normalize_text(source_text)
        normalized_target = self.normalize_text(target_text)

        return {
            self.translation_key: {
                self.source_field: normalized_source,
                self.target_field: normalized_target,
            }
        }
