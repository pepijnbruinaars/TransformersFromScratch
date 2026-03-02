"""Training entry point with RunPod Spot instance support.

Supports both encoder-decoder (translation) and decoder-only (generative) models.

Usage:
    # Fresh training
    python train.py --config configs/experiment_config.yaml

    # Resume from latest checkpoint
    python train.py --resume

    # Resume from specific checkpoint
    python train.py --checkpoint /path/to/checkpoint.pt
"""
import argparse
from dataclasses import replace
from functools import partial
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.loader import ConfigLoader
from src.config.environment import EnvironmentConfig
from src.config.base import CheckpointConfig
from src.constants import PAD_TOKEN
from src.models.Transformer import Transformer
from src.models.DecoderOnlyTransformer import DecoderOnlyTransformer
from src.data import (
    DataPipelineConfig,
    EuroParlDataProvider,
    create_dataloaders_from_config,
)
from src.data.generative_dataset import GenerativeDataset, generative_collate_fn
from src.data.providers import PROVIDER_REGISTRY
from src.data.providers.multi_corpus import MultiCorpusDataProvider
from src.tokenization.tokenizer import CustomTokenizer
from src.training.trainer import Trainer
from src.training.generation import GenerationSampler

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train transformer model with RunPod Spot instance support"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to resume from",
    )
    return parser.parse_args()


def _create_generative_dataloaders(
    provider_config,
    tokenizer: CustomTokenizer,
    split_config,
    preprocessing_config,
    train_loader_config,
    val_loader_config,
):
    """Create dataloaders for generative task."""
    from datasets import load_dataset

    logger.info("Loading generative dataset...")

    # Load dataset with proper split handling
    try:
        # Try loading with split=None to get all splits
        dataset = load_dataset(
            provider_config.dataset_name,
            provider_config.dataset_config,
            split=None,
        )
        train_dataset_hf = dataset["train"]
        val_dataset_hf = dataset.get("validation", dataset.get("test"))
    except (ValueError, KeyError):
        # If split=None fails, load the train split and split it ourselves
        logger.info(f"Loading dataset without predefined splits. Creating val split from config ratios.")
        dataset = load_dataset(
            provider_config.dataset_name,
            provider_config.dataset_config,
            split=provider_config.split,
        )

        # Split the train dataset into train and val based on config
        train_ratio = split_config.train
        split_dataset = dataset.train_test_split(
            test_size=(1.0 - train_ratio),
            seed=42
        )
        train_dataset_hf = split_dataset["train"]
        val_dataset_hf = split_dataset["test"]

    # Create datasets
    train_dataset = GenerativeDataset(
        train_dataset_hf,
        tokenizer,
        preprocessing_config.sequence_config.max_length,
        preprocessing_config.generative_config.text_field,
    )

    val_dataset = GenerativeDataset(
        val_dataset_hf,
        tokenizer,
        preprocessing_config.sequence_config.max_length,
        preprocessing_config.generative_config.text_field,
    )

    # Create dataloaders
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
    collate_fn = partial(generative_collate_fn, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_loader_config.batch_size,
        shuffle=train_loader_config.shuffle,
        collate_fn=collate_fn,
        num_workers=train_loader_config.num_workers,
        pin_memory=train_loader_config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_loader_config.batch_size,
        shuffle=val_loader_config.shuffle,
        collate_fn=collate_fn,
        num_workers=val_loader_config.num_workers,
        pin_memory=val_loader_config.pin_memory,
    )

    return train_loader, val_loader, None


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load experiment configuration
    config_loader = ConfigLoader()
    experiment_config = config_loader.from_yaml(args.config)
    logger.info(f"Loaded experiment configuration: {experiment_config.experiment_name}")

    # Detect model architecture
    architecture = experiment_config.model_config.architecture
    logger.info(f"Model architecture: {architecture}")

    # Log batch size and gradient accumulation info
    batch_size = experiment_config.data_config.batch_size
    accumulation_steps = experiment_config.training_config.optimizer_config.accumulation_steps
    effective_batch_size = batch_size * accumulation_steps
    logger.info(f"Using batch_size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"Effective batch_size: {effective_batch_size}")

    # Initialize environment config for path resolution
    runpod_base_path = None
    if experiment_config.runpod_config and experiment_config.runpod_config.enabled:
        runpod_base_path = experiment_config.runpod_config.base_path
        logger.info(f"RunPod mode enabled with base path: {runpod_base_path}")

    env_config = EnvironmentConfig(base_path=runpod_base_path)

    # Resolve paths using environment config
    tokenizer_path = env_config.get_tokenizer_path(
        experiment_config.data_config.tokenizer_path
    )
    checkpoint_dir = env_config.get_checkpoint_dir(
        experiment_config.checkpoint_config.save_dir
    )
    log_dir = env_config.get_log_dir(
        experiment_config.training_config.tensorboard_log_dir
    )

    logger.info(f"Resolved paths:")
    logger.info(f"  - Tokenizer: {tokenizer_path}")
    logger.info(f"  - Checkpoints: {checkpoint_dir}")
    logger.info(f"  - Logs: {log_dir}")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create modified configs with resolved paths
    resolved_checkpoint_config = CheckpointConfig(
        save_dir=checkpoint_dir,
        save_frequency=experiment_config.checkpoint_config.save_frequency,
    )

    resolved_training_config = replace(
        experiment_config.training_config,
        tensorboard_log_dir=log_dir,
    )

    # ==================== GENERATIVE MODE ====================
    if architecture == "decoder_only":
        logger.info("Training decoder-only (generative) model")

        # Extract generative config
        gen_config = experiment_config.generative_config

        # Load and train tokenizer
        logger.info(f"Initializing tokenizer from: {tokenizer_path}")
        tokenizer = CustomTokenizer(
            tokenizer_path,
            vocab_size=experiment_config.data_config.vocab_size,
        )

        if not tokenizer.trained:
            logger.info("Loading dataset to train tokenizer...")
            from datasets import load_dataset

            dataset = load_dataset(
                gen_config.dataset.dataset_name,
                gen_config.dataset.dataset_config,
                split=gen_config.dataset.split,
            )

            # Extract text samples for tokenizer training
            text_samples = [item[gen_config.text_field] for item in dataset]
            logger.info(f"Training tokenizer on {len(text_samples)} samples...")
            tokenizer.train(text_samples)
            tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to: {tokenizer_path}")

        # Create dataloaders for generative task
        train_loader, val_loader, _ = _create_generative_dataloaders(
            provider_config=gen_config.dataset,
            tokenizer=tokenizer,
            split_config=gen_config.split,
            preprocessing_config=experiment_config.generative_preprocessing_config,
            train_loader_config=gen_config.train_loader,
            val_loader_config=gen_config.val_loader,
        )

        # Create decoder-only model
        model = DecoderOnlyTransformer(
            n_blocks=experiment_config.model_config.n_block,
            d_model=experiment_config.model_config.d_model,
            d_ff=experiment_config.model_config.d_ff,
            n_heads=experiment_config.model_config.n_head,
            dropout=experiment_config.model_config.dropout_rate,
            vocab_size=tokenizer.vocabulary_size,
            sequence_length=experiment_config.model_config.sequence_length,
            use_flash_attention=experiment_config.model_config.use_flash_attention,
            activation=experiment_config.model_config.activation,
            use_rope=experiment_config.model_config.use_rope,
        )

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tokenizer=tokenizer,
            training_config=resolved_training_config,
            checkpoint_config=resolved_checkpoint_config,
            experiment_name=experiment_config.experiment_name,
            runpod_config=experiment_config.runpod_config,
            model_type="decoder_only",
            generation_config=experiment_config.generation_config,
        )

        # Setup generation sampler if enabled
        if hasattr(experiment_config, 'generation_config') and experiment_config.generation_config.enabled:
            gen_sampler = GenerationSampler(
                model=model,
                tokenizer=tokenizer,
                prompts=experiment_config.generation_config.prompts,
                temperatures=experiment_config.generation_config.temperatures,
                max_new_tokens=experiment_config.generation_config.max_new_tokens,
            )
            # Store sampler for use during training
            trainer.generation_sampler = gen_sampler

    # ==================== TRANSLATION MODE ====================
    else:
        logger.info("Training encoder-decoder (translation) model")

        # Check if multi-corpus config exists
        use_multi_corpus = (
            experiment_config.multi_corpus_config is not None
        )

        if use_multi_corpus:
            logger.info("Using multi-corpus configuration")
            multi_corpus_config = experiment_config.multi_corpus_config

            # Create multi-corpus provider
            provider = MultiCorpusDataProvider(
                config=multi_corpus_config,
                provider_registry=PROVIDER_REGISTRY,
            )

            data_config = multi_corpus_config
        else:
            logger.info("Using single-corpus configuration (legacy mode)")
            # Legacy single-corpus mode
            data_config = DataPipelineConfig.create_default(
                provider_name="europarl",
                max_sequence_length=512,
                batch_size=experiment_config.data_config.batch_size,
            )
            provider = EuroParlDataProvider(data_config.provider_config)

        # Load raw data to train tokenizer
        if experiment_config.training_config.logging_verbosity > 0:
            logger.info("Loading raw data for tokenizer training...")
        splits = provider.load(data_config.split_config)

        # Extract sentences from training data for tokenizer training
        def extract_sentences(dataset):
            """Extract all sentences from dataset."""
            sentences = []
            for item in tqdm(
                dataset,
                desc="Extracting sentences",
                disable=experiment_config.training_config.logging_verbosity == 0,
            ):
                translation = item["translation"]
                sentences.append(translation["en"])
                sentences.append(translation["nl"])
            return sentences

        train_sentences = extract_sentences(splits["train"])

        # Train or load tokenizer
        logger.info(f"Initializing tokenizer from: {tokenizer_path}")
        tokenizer = CustomTokenizer(
            tokenizer_path,
            vocab_size=experiment_config.data_config.vocab_size,
        )

        if not tokenizer.trained:
            logger.info(f"Training tokenizer on {len(train_sentences)} sentences...")
            tokenizer.train(train_sentences)
            tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to: {tokenizer_path}")

        train_loader, val_loader, test_loader = create_dataloaders_from_config(
            provider=provider,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer,
            config=data_config,
        )

        trainer = Trainer(
            model=Transformer(
                n_blocks=experiment_config.model_config.n_block,
                n_heads=experiment_config.model_config.n_head,
                d_model=experiment_config.model_config.d_model,
                d_ff=experiment_config.model_config.d_ff,
                dropout=experiment_config.model_config.dropout_rate,
                source_length=512,
                target_length=512,
                source_vocabulary_size=tokenizer.vocabulary_size,
                target_vocabulary_size=tokenizer.vocabulary_size,
                activation=experiment_config.model_config.activation,
            ),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tokenizer=tokenizer,
            training_config=resolved_training_config,
            checkpoint_config=resolved_checkpoint_config,
            experiment_name=experiment_config.experiment_name,
            runpod_config=experiment_config.runpod_config,
            model_type="encoder_decoder",
            generation_config=experiment_config.generation_config,
        )

        trainer.load_sample_sentences()

    # Train or resume based on arguments
    num_epochs = experiment_config.training_config.num_epochs

    if args.checkpoint:
        # Resume from specific checkpoint
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.resume(num_epochs=num_epochs, checkpoint_path=args.checkpoint)
    elif args.resume:
        # Resume from latest checkpoint
        logger.info("Resuming from latest checkpoint...")
        trainer.resume(num_epochs=num_epochs)
    else:
        # Fresh training
        logger.info("Starting fresh training...")
        trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
