from datetime import datetime
import logging

from tqdm import tqdm
from src.config.loader import ConfigLoader
from src.models.Transformer import Transformer
from src.data import (
    DataPipelineConfig,
    EuroParlDataProvider,
    create_dataloaders_from_config,
)
from torch.utils.tensorboard.writer import SummaryWriter

from src.tokenization.tokenizer import CustomTokenizer
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_config = DataPipelineConfig.create_default(
        provider_name="europarl",
        max_sequence_length=512,
        batch_size=8,
    )
    
    config_loader = ConfigLoader()
    experiment_config = config_loader.from_yaml("configs/experiment_config.yaml")
    logger.info(f"Loaded experiment configuration: {experiment_config.experiment_name}")
    
    provider = EuroParlDataProvider(data_config.provider_config)
    
    # Load raw data to train tokenizer
    if experiment_config.training_config.logging_verbosity > 0:
        logger.info("Loading raw data for tokenizer training...")
    splits = provider.load(data_config.split_config)
    
    # Extract sentences from training data for tokenizer training
    def extract_sentences(dataset):
        """Extract all sentences from dataset."""
        sentences = []
        for item in tqdm(dataset, desc="Extracting sentences", disable=experiment_config.training_config.logging_verbosity == 0):
            translation = item["translation"]
            sentences.append(translation["en"])
            sentences.append(translation["nl"])
        return sentences
    
    train_sentences = extract_sentences(splits["train"])
    
    # Train or load tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = CustomTokenizer("models/tokenizers/europarl_tokenizer.json")
    
    if not tokenizer.trained:
        logger.info(f"Training tokenizer on {len(train_sentences)} sentences...")
        tokenizer.train(train_sentences)
        tokenizer.save("models/tokenizers/europarl_tokenizer.json")
    
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
        training_config=experiment_config.training_config,
        checkpoint_config=experiment_config.checkpoint_config,
        experiment_name=experiment_config.experiment_name,
    )
    
    trainer.train(num_epochs=experiment_config.training_config.num_epochs)

if __name__ == "__main__":
    main()