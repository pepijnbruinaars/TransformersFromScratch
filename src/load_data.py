from datasets import load_dataset, Dataset  # type: ignore
from typing import Dict, Any, Tuple
from torch.utils.data import random_split, Subset

from tokenizer import CustomTokenizer


def get_max_sequence_length(
    dataset: Subset,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
) -> Tuple[int, int]:
    """Determines the maximum sequence length for source and target sentences in the dataset.

    Args:
        dataset: The dataset containing the sentences.
        source_tokenizer (CustomTokenizer): The tokenizer for the source language.
        target_tokenizer (CustomTokenizer): The tokenizer for the target language.

    Returns:
        Tuple[int, int]: A tuple containing the maximum source and target sequence lengths respectively.
    """
    max_source_length = 0
    max_target_length = 0

    for item in dataset:  # type: ignore
        source_length = len(source_tokenizer.encode(item["translation"]["nl"]))  # type: ignore
        target_length = len(target_tokenizer.encode(item["translation"]["nl"]))  # type: ignore

        max_source_length = max(max_source_length, source_length)
        max_target_length = max(max_target_length, target_length)

    # Return the maximum of source and target lengths
    return max_source_length, max_target_length


def get_sentences_from_data(dataset) -> tuple[object, list[str], list[str]]:
    """Loads the OPUS Books dataset for English to Dutch translation."""
    # Load the dataset
    dataset = load_dataset("opus_books", "en-nl", split="train")

    source_sentences: list[str] = []
    target_sentences: list[str] = []

    # Iterate through the dataset and extract the English and Dutch translations
    # Each item has a "translation" field with a dictionary containing "en" and "nl" key-translations
    for item in dataset:
        # Convert the item to a dictionary to avoid type errors
        item_dict: Dict[str, Any] = dict(item)

        source_sentences.append(item_dict["translation"]["en"])
        target_sentences.append(item_dict["translation"]["nl"])
    return dataset, source_sentences, target_sentences


def load_opus_data(split: float = 0.8, deterministic: bool = False):
    """Get a split of the OPUS Books dataset for English to Dutch translation.

    Args:
        split (float, optional): The percentage of the dataset to use for training. Defaults to 0.8.

    Returns:
        _type_: A tuple containing the training and validation datasets.
    """
    # Keep split percentage of the dataset for training, other part for validation
    dataset: Dataset = load_dataset("opus_books", "en-nl", split="train")  # type: ignore

    seed = 42 if deterministic else None  # Use a fixed seed for reproducibility
    dataset = dataset.shuffle(seed=seed)  # Shuffle the dataset for randomness

    # Calculate the sizes for training and validation sets
    dataset_length = len(dataset)  # type: ignore # Fixed typo in variable name
    train_size = int(dataset_length * split)
    validation_size = dataset_length - train_size

    train_data, val_data = random_split(dataset, [train_size, validation_size])  # type: ignore

    return (
        train_data,
        val_data,
    )
