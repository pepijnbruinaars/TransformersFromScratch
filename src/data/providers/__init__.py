"""Provider registry for multi-corpus support."""

from typing import Type, Dict
from .base import DataProvider
from .euro_parl import EuroParlDataProvider
from .opus_books import OpusBooksDataProvider
from .wikimedia import WikimediaDataProvider
from .open_subtitles import OpenSubtitlesDataProvider

# Registry mapping provider names to classes
PROVIDER_REGISTRY: Dict[str, Type[DataProvider]] = {
    "europarl": EuroParlDataProvider,
    "opus_books": OpusBooksDataProvider,
    "wikimedia": WikimediaDataProvider,
    "open_subtitles": OpenSubtitlesDataProvider,
}


def register_provider(name: str, provider_class: Type[DataProvider]) -> None:
    """Register a new provider in the registry.

    Args:
        name: Provider name (used in config)
        provider_class: Provider class
    """
    PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> Type[DataProvider]:
    """Get a provider class by name.

    Args:
        name: Provider name

    Returns:
        Provider class

    Raises:
        ValueError: If provider not found
    """
    if name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Provider '{name}' not found. Available: {list(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[name]


__all__ = [
    "DataProvider",
    "EuroParlDataProvider",
    "OpusBooksDataProvider",
    "WikimediaDataProvider",
    "OpenSubtitlesDataProvider",
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_provider_class",
]
