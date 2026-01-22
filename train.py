"""Training entry point with RunPod Spot instance support.

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
import logging
import os

from tqdm import tqdm

from src.config.loader import ConfigLoader
from src.config.environment import EnvironmentConfig
from src.config.base import CheckpointConfig
from src.models.Transformer import Transformer
from src.data import (
    DataPipelineConfig,
    EuroParlDataProvider,
    create_dataloaders_from_config,
)
from src.tokenization.tokenizer import CustomTokenizer
from src.training.trainer import Trainer

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
    logger.info(f"Using batch_size: {experiment_config.data_config.batch_size}")

    # Initialize environment config for path resolution
    # If RunPod config exists and is enabled, use its base_path
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

    # Update training config with resolved log dir
    resolved_training_config = replace(
        experiment_config.training_config,
        tensorboard_log_dir=log_dir,
    )

    # Create data config with batch_size from experiment config
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
    tokenizer = CustomTokenizer(tokenizer_path)

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
        ),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        training_config=resolved_training_config,
        checkpoint_config=resolved_checkpoint_config,
        experiment_name=experiment_config.experiment_name,
        runpod_config=experiment_config.runpod_config,
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
