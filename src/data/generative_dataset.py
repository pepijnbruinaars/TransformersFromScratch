"""PyTorch Dataset for generative/causal language modeling with variable-length sequences."""

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from typing import Dict
import logging

import torch

from ..tokenization.tokenizer import CustomTokenizer
from ..constants import PAD_TOKEN

logger = logging.getLogger(__name__)


class GenerativeDataset(TorchDataset):
    """Dataset for causal language modeling (next-token prediction).

    Features:
    - Variable-length sequences (no fixed padding)
    - Token shifting: input=[t0, t1, t2, ...], target=[t1, t2, t3, ...]
    - Returns sequences up to max_length
    - Padding handled by custom collate_fn in DataLoader
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: CustomTokenizer,
        max_length: int,
        text_field: str = "text",
    ) -> None:
        """Initialize generative dataset.

        Args:
            dataset: HuggingFace Dataset with text data
            tokenizer: CustomTokenizer instance
            max_length: Maximum sequence length (truncate if longer)
            text_field: Field name containing text (default: "text")
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

        logger.info(
            f"Initialized GenerativeDataset with {len(dataset)} examples, "
            f"max_length={max_length}, text_field='{text_field}', pad_token_id={self.pad_token_id}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example.

        Returns:
            Dict with:
            - input_ids: tokens[:-1] (what the model sees)
            - target_ids: tokens[1:] (what the model predicts)
            - length: actual sequence length (before padding)
        """
        # Get raw text
        text = self.dataset[idx][self.text_field]

        # Tokenize (returns token IDs)
        token_ids = self.tokenizer.encode(text)

        # Truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Need at least 2 tokens to create input/target pair
        if len(token_ids) < 2:
            # Pad to minimum 2 tokens
            token_ids = token_ids + [self.pad_token_id] * (2 - len(token_ids))

        # Shift tokens: input is t[0:-1], target is t[1:]
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "length": len(input_ids),  # Actual length (before padding)
        }


def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask (lower triangular).

    Prevents attention to future tokens. Shape: [seq_length, seq_length]

    Args:
        seq_length: Sequence length
        device: Device to create mask on

    Returns:
        Lower triangular matrix (True = attend, False = don't attend)
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
    return mask


def generative_collate_fn(
    batch: list,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences.

    Pads sequences to the longest in the batch and generates causal masks.

    Args:
        batch: List of dicts from GenerativeDataset
        pad_token_id: Padding token ID from tokenizer

    Returns:
        Dict with:
        - input_ids: [batch_size, max_seq_len_in_batch]
        - target_ids: [batch_size, max_seq_len_in_batch]
        - attention_mask: [batch_size, max_seq_len_in_batch] (1=real, 0=padding)
        - causal_mask: [max_seq_len_in_batch, max_seq_len_in_batch] (shared for whole batch)
        - lengths: [batch_size] (original lengths before padding)
    """
    # Get the longest sequence in this batch
    max_len = max(item["length"] for item in batch)

    padded_input_ids = []
    padded_target_ids = []
    attention_masks = []
    lengths = []

    # Pad each sequence to max_len
    for item in batch:
        input_ids = item["input_ids"]
        target_ids = item["target_ids"]
        length = item["length"]

        # Pad input_ids
        padding_len = max_len - len(input_ids)
        padded_input = torch.nn.functional.pad(
            input_ids, (0, padding_len), value=pad_token_id
        )
        padded_input_ids.append(padded_input)

        # Pad target_ids
        padded_target = torch.nn.functional.pad(
            target_ids, (0, padding_len), value=pad_token_id
        )
        padded_target_ids.append(padded_target)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(max_len, dtype=torch.bool)
        attention_mask[length:] = False
        attention_masks.append(attention_mask)

        lengths.append(length)

    # Stack into batches
    batch_input_ids = torch.stack(padded_input_ids)
    batch_target_ids = torch.stack(padded_target_ids)
    batch_attention_mask = torch.stack(attention_masks)
    batch_lengths = torch.tensor(lengths, dtype=torch.long)

    # Create causal mask (same for all sequences in batch)
    causal_mask = create_causal_mask(max_len, batch_input_ids.device)

    return {
        "input_ids": batch_input_ids,
        "target_ids": batch_target_ids,
        "attention_mask": batch_attention_mask,
        "causal_mask": causal_mask,
        "lengths": batch_lengths,
    }