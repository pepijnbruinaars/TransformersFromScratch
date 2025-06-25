from load_data import get_max_sequence_length, load_opus_data, get_sentences_from_data
from tokenizer import CustomTokenizer
from transformer import Transformer
from dataset import CustomDataset
from torch.utils.data import DataLoader
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

    # Create datasets for training and validation
    train = CustomDataset(
        train_raw,  # type: ignore
        source_tokenizer=dutch_tokenizer,
        target_tokenizer=english_tokenizer,
        sequence_length=100,
    )
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=dutch_tokenizer,
        target_tokenizer=english_tokenizer,
        sequence_length=100,
    )

    source_length, target_length = get_max_sequence_length(
        full_dataset[0], english_tokenizer, dutch_tokenizer
    )

    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=32, shuffle=False)

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
        source_length=source_length,
        target_length=target_length,
        source_vocabulary_size=english_tokenizer.vocabulary_size,
        target_vocabulary_size=dutch_tokenizer.vocabulary_size,
    )
    print(transformer)

    ########################
    ## TRAINING THE MODEL ##
    ########################
    device = get_device()
    transformer.to(device)
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
