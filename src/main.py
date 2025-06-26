from datetime import datetime
from pathlib import Path

from tqdm import tqdm  # type: ignore
from constants import PAD_TOKEN
from load_data import get_max_sequence_length, load_opus_data, get_sentences_from_data
from tokenizer import CustomTokenizer
from transformer import Transformer
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


def get_device() -> str:
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_or_load_tokenizers(
    dutch_sentences: list[str], english_sentences: list[str]
) -> tuple[CustomTokenizer, CustomTokenizer]:
    # Creating tokenizers
    dutch_tokenizer = CustomTokenizer("models/tokenizers/dutch_tokenizer.json")
    english_tokenizer = CustomTokenizer("models/tokenizers/english_tokenizer.json")

    if not dutch_tokenizer.trained:
        print("Training Dutch tokenizer...")
        dutch_tokenizer.train(dutch_sentences)
        dutch_tokenizer.save("models/tokenizers/dutch_tokenizer.json")
    if not english_tokenizer.trained:
        print("Training English tokenizer...")
        english_tokenizer.train(english_sentences)
        english_tokenizer.save("models/tokenizers/english_tokenizer.json")

    # Testing the tokenizers
    print("Testing Dutch tokenizer...")
    sample_dutch_sentences = dutch_sentences[:20]
    for sentence in sample_dutch_sentences:
        dutch_tokenizer.print_tokens(sentence)

    # Testing the English tokenizer
    print("Testing English tokenizer...")
    sample_english_sentences = english_sentences[:20]
    for sentence in sample_english_sentences:
        english_tokenizer.print_tokens(sentence)

    return dutch_tokenizer, english_tokenizer


def train_model(
    transformer: Transformer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    nr_epochs: int = 10,
    learning_rate: float = 1e-4,
) -> None:
    """Train the transformer model."""
    device = get_device()
    print(f"Using device: {device}")
    transformer.to(device)

    # Check model folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"models/transformer/{timestamp}"
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/transformer").mkdir(parents=True, exist_ok=True)

    # TensorBoard to visualize training
    writer = SummaryWriter(log_dir=model_folder)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, eps=1e-9)

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.token_to_id(PAD_TOKEN),
        label_smoothing=0.1,
    ).to(device)

    iterations = 0
    for epoch in range(nr_epochs):
        transformer.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{nr_epochs}")
        for batch in batch_iterator:
            source, target, source_mask, target_mask, label = (
                batch["source"].to(device),
                batch["target"].to(device),
                batch["source_mask"].to(device),
                batch["target_mask"].to(device),
                batch["label"].to(device),
            )

            # Forward pass
            encoder_output = transformer.encode(source, source_mask)
            decoder_output = transformer.decode(
                target, encoder_output, source_mask, target_mask
            )
            projection = transformer.project(decoder_output)

            # Compute loss
            loss = loss_function(
                projection.view(-1, target_tokenizer.vocabulary_size), label.view(-1)
            )

            # Update progress bar and tensorboard
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Loss/train", loss.item(), iterations)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            iterations += 1

        # Save the model after each epoch
        model_path = f"{model_folder}/transformer_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model_state_dict": transformer.state_dict(),
                "epoch": nr_epochs,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_path,
        )
        print(f"Model saved to {model_path}")

    # Save the final model
    final_model_path = f"{model_folder}/transformer_final.pt"
    torch.save(
        {
            "model_state_dict": transformer.state_dict(),
            "epoch": nr_epochs,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )


def main() -> None:
    ######################
    ## LOADING THE DATA ##
    ######################
    full_dataset = load_opus_data(1.0)
    _, dutch_sentences, english_sentences = get_sentences_from_data(full_dataset)

    dutch_tokenizer, english_tokenizer = train_or_load_tokenizers(
        dutch_sentences, english_sentences
    )

    train_raw, validation_raw = load_opus_data(0.8)

    source_length, target_length = get_max_sequence_length(
        full_dataset[0], english_tokenizer, dutch_tokenizer
    )

    # Use the maximum of both for consistent sequence length
    max_sequence_length = max(source_length, target_length)

    # Create datasets for training and validation
    train = CustomDataset(
        train_raw,  # type: ignore
        source_tokenizer=dutch_tokenizer,
        target_tokenizer=english_tokenizer,
        sequence_length=max_sequence_length,
    )
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=dutch_tokenizer,
        target_tokenizer=english_tokenizer,
        sequence_length=max_sequence_length,
    )

    train_dataloader = DataLoader(train, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=8, shuffle=False)

    ########################
    ## BUILDING THE MODEL ##
    ########################

    # Model parameters from the paper
    n_blocks = 6
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1
    transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        source_length=max_sequence_length,
        target_length=max_sequence_length,
        source_vocabulary_size=english_tokenizer.vocabulary_size,
        target_vocabulary_size=dutch_tokenizer.vocabulary_size,
    )
    print(transformer)

    ########################
    ## TRAINING THE MODEL ##
    ########################
    train_model(
        transformer=transformer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        source_tokenizer=english_tokenizer,
        target_tokenizer=dutch_tokenizer,
    )


if __name__ == "__main__":
    main()
