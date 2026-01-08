from typing import Optional, Tuple
import torch
import torch.nn as nn

import math


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that calculates the scaled dot product attention for a set of
    Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
    Args:
        Q (FloatTensor): The query vector
        K (FloatTensor): The keys vector
        V (FloatTensor): The values vector
    """
    # 1. Calculate scores and normalize
    scores = torch.matmul(Q, K.transpose(-2, -1))
    normalized_scores = scores / math.sqrt(K.shape[-1])
    if mask is not None:
        # Ensure mask can be broadcasted to match the scores tensor shape
        normalized_scores = normalized_scores.masked_fill(mask == 0, float("-inf"))

    # 2. Apply softmax to get probabilities
    probabilities = torch.softmax(normalized_scores, dim=-1)
    if dropout is not None:
        probabilities = dropout(probabilities)

    output = torch.matmul(probabilities, V)

    return output, probabilities


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention block used in the encoder and decoder part of the transformer.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()

        # Assert dimensions
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Query, key and value weights
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output weights and dropout
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.FloatTensor,
        K: torch.FloatTensor,
        V: torch.FloatTensor,
        mask: torch.Tensor,
    ):
        batch_size = Q.shape[0]
        
        # Get Q', K', and V'
        query: torch.Tensor = self.W_Q(Q)
        key: torch.Tensor = self.W_K(K)
        value: torch.Tensor = self.W_V(V)

        # Reshape to multi-head format: (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        query = query.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        x, attention_scores = scaled_dot_product_attention(
            query,
            key,
            value,
            mask,
            self.dropout,
        )

        # Reshape back: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.W_O(x)
