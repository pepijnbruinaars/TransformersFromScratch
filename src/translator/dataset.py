import torch
from datasets import Dataset  # type: ignore
from torch.utils.data import Dataset as TorchDataset
from typing import TypedDict

from ..constants import END_TOKEN, PAD_TOKEN, START_TOKEN
from ..tokenization.tokenizer import CustomTokenizer


class DatasetPair(TypedDict):
    """Typed dictionary for dataset output."""

    source: torch.Tensor
    target: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    label: torch.Tensor
    source_text: str
    target_text: str


def attention_mask(size: int) -> torch.Tensor:
    """_summary_

    Args:
        size (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  # Mask out the upper triangular part


class CustomDataset(TorchDataset):
    """Custom dataset for handling tokenized sentences."""

    def __init__(
        self,
        dataset: Dataset,
        source_tokenizer: CustomTokenizer,
        target_tokenizer: CustomTokenizer,
        sequence_length: int,
    ) -> None:
        """
        Initialize the dataset with source and target sentences.

        """
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.sequence_length = sequence_length

        # Source-side special tokens
        self.source_start_token = torch.tensor(
            source_tokenizer.token_to_id(START_TOKEN), dtype=torch.int64
        )
        self.source_end_token = torch.tensor(
            source_tokenizer.token_to_id(END_TOKEN), dtype=torch.int64
        )
        self.source_padding_token = torch.tensor(
            source_tokenizer.token_to_id(PAD_TOKEN), dtype=torch.int64
        )

        # Target-side special tokens
        self.target_start_token = torch.tensor(
            target_tokenizer.token_to_id(START_TOKEN), dtype=torch.int64
        )
        self.target_end_token = torch.tensor(
            target_tokenizer.token_to_id(END_TOKEN), dtype=torch.int64
        )
        self.target_padding_token = torch.tensor(
            target_tokenizer.token_to_id(PAD_TOKEN), dtype=torch.int64
        )
        
        # Pre-compute target causal mask (same for all sequences of this length)
        self._target_causal_mask = attention_mask(sequence_length)

    def _mask(self, tensor: torch.Tensor, padding_token: torch.Tensor) -> torch.Tensor:
        """Create a mask for the tensor, where padding tokens are masked out."""
        return (tensor != padding_token).unsqueeze(0).unsqueeze(0).int()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> DatasetPair:
        # Unpack the pair from the dataset
        pair = self.dataset[index]
        source_text = pair["translation"]["en"]
        target_text = pair["translation"]["nl"]

        # Tokenize the source and target sentences
        source_tokens = self.source_tokenizer.encode(source_text)
        target_tokens = self.target_tokenizer.encode(target_text)

        # Add start and end tokens to the source tensor
        source_tensor = torch.tensor(
            [self.source_start_token] + source_tokens + [self.source_end_token],
            dtype=torch.int64,
        )
        # Add start token to the target tensor (NO end token for target)
        target_tensor = torch.tensor(
            [self.target_start_token] + target_tokens, dtype=torch.int64
        )

        # The label tensor is the target tokens with an end token appended
        # This is because the model predicts the next token in the sequence, so we expect the model to predict the end token as well
        label_tensor = torch.tensor(
            target_tokens + [self.target_end_token], dtype=torch.int64
        )

        # Truncate sequences if they exceed the maximum length
        if len(source_tensor) > self.sequence_length:
            source_tensor = source_tensor[: self.sequence_length]
        if len(target_tensor) > self.sequence_length:
            target_tensor = target_tensor[: self.sequence_length]
        if len(label_tensor) > self.sequence_length:
            label_tensor = label_tensor[: self.sequence_length]

        # Pad the source and target tensors to the specified sequence length
        number_padding_source = self.sequence_length - len(source_tensor)
        number_padding_target = self.sequence_length - len(target_tensor)

        # Concatenate padding tokens to the source and target tensors
        source_tensor = torch.cat(
            [
                source_tensor,
                self.source_padding_token.repeat(number_padding_source),
            ]
        )
        target_tensor = torch.cat(
            [
                target_tensor,
                self.target_padding_token.repeat(number_padding_target),
            ]
        )

        label_tensor = torch.cat(
            [
                label_tensor,
                self.target_padding_token.repeat(number_padding_target),
            ]
        )

        assert (
            source_tensor.size(0) == self.sequence_length
        ), f"Source tensor length: {len(source_tensor)} - expected {self.sequence_length}."
        assert (
            target_tensor.size(0) == self.sequence_length
        ), f"Target tensor length: {len(target_tensor)} - expected {self.sequence_length}."
        assert (
            label_tensor.size(0) == self.sequence_length
        ), f"Label tensor length: {len(label_tensor)} - expected {self.sequence_length}."

        # Apply masks to the source and target tensors
        masked_source_tensor = self._mask(source_tensor, self.source_padding_token)
        masked_target_tensor = self._mask(target_tensor, self.target_padding_token) & self._target_causal_mask

        return {
            "source": source_tensor,
            "target": target_tensor,
            "source_mask": masked_source_tensor,
            "target_mask": masked_target_tensor,
            "label": label_tensor,
            "source_text": source_text,
            "target_text": target_text,
        }
