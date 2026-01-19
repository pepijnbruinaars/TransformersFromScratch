"""Example usage of the refactored data pipeline.

This example demonstrates how to use the new data pipeline with:
1. Configuration-driven setup
2. Multiple data providers
3. Type-safe interfaces
"""

import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    DataPipelineConfig,
    DataProviderConfig,
    DatasetSplitConfig,
    SequenceConfig,
    TranslationConfig,
    PreprocessingConfig,
    DataLoaderConfig,
    WikimediaDataProvider,
    EuroParlDataProvider,
    OpusBooksDataProvider,
    create_dataloaders_from_config,
    compute_optimal_max_length,
)
from src.tokenization.tokenizer import CustomTokenizer
from src.constants import PAD_TOKEN, START_TOKEN, END_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_usage():
    """Example 1: Simple usage with default configuration."""
    logger.info("=" * 80)
    logger.info("Example 1: Simple usage with default configuration")
    logger.info("=" * 80)

    # Create a simple configuration
    config = DataPipelineConfig.create_default(
        provider_name="europarl",
        max_sequence_length=128,
        batch_size=16,
    )

    # Initialize provider
    provider = EuroParlDataProvider(config.provider_config)

    # Load or train tokenizer
    tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")
    if not tokenizer.trained:
        logger.info("Training tokenizer...")
        raw_dataset = provider.load_raw()
        source_sentences, target_sentences = provider.get_all_sentences(raw_dataset)
        combined_sentences = source_sentences + target_sentences
        tokenizer.train(combined_sentences)
        tokenizer.save("models/tokenizers/shared_tokenizer.json")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_config(
        provider=provider,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        config=config,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Inspect a batch
    batch = next(iter(train_loader))
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Source shape: {batch['source'].shape}")
    logger.info(f"Target shape: {batch['target'].shape}")
    logger.info(f"Source mask shape: {batch['source_mask'].shape}")
    logger.info(f"Target mask shape: {batch['target_mask'].shape}")


def example_2_custom_configuration():
    """Example 2: Custom configuration with specific settings."""
    logger.info("=" * 80)
    logger.info("Example 2: Custom configuration with specific settings")
    logger.info("=" * 80)

    # Create custom configuration
    config = DataPipelineConfig(
        provider_config=DataProviderConfig(
            name="wikimedia",
            deterministic=True,
            seed=42,
        ),
        split_config=DatasetSplitConfig(
            train=0.8,
            val=0.1,
            test=0.1,
            max_train_size=100_000,  # Cap training set at 100k
        ),
        preprocessing_config=PreprocessingConfig(
            sequence_config=SequenceConfig(
                max_length=256,
                truncation=True,
                padding="max_length",
            ),
            translation_config=TranslationConfig(
                source_lang="en",
                target_lang="nl",
                source_field="en",
                target_field="nl",
            ),
            add_special_tokens=True,
            return_attention_mask=True,
        ),
        train_loader_config=DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        val_loader_config=DataLoaderConfig(
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    )

    # Initialize provider
    provider = WikimediaDataProvider(config.provider_config)

    # Load tokenizer
    tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_config(
        provider=provider,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        config=config,
    )

    logger.info(f"Created dataloaders with batch_size={config.train_loader_config.batch_size}")


def example_3_multiple_providers():
    """Example 3: Using different data providers."""
    logger.info("=" * 80)
    logger.info("Example 3: Using different data providers")
    logger.info("=" * 80)

    tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")

    # EuroParl provider
    europarl_config = DataProviderConfig(name="europarl")
    europarl_provider = EuroParlDataProvider(europarl_config)
    logger.info(f"EuroParl provider: {europarl_provider.__class__.__name__}")

    # OPUS Books provider
    opus_config = DataProviderConfig(name="opus_books")
    opus_provider = OpusBooksDataProvider(opus_config)
    logger.info(f"OPUS Books provider: {opus_provider.__class__.__name__}")

    # Wikimedia provider (local files)
    wikimedia_config = DataProviderConfig(name="wikimedia")
    wikimedia_provider = WikimediaDataProvider(wikimedia_config)
    logger.info(f"Wikimedia provider: {wikimedia_provider.__class__.__name__}")


if __name__ == "__main__":
    # Run examples
    try:
        example_1_simple_usage()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    try:
        example_2_custom_configuration()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    try:
        example_3_multiple_providers()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")