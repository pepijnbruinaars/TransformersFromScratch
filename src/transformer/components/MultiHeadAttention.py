import torch
import torch.nn as nn

import math


def scaled_dot_product_attention(
    Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor
):
    """Function that calculates the scaled dot product attention for a set of

    Args:
        Q (FloatTensor): The query vector
        K (FloatTensor): The keys vector
        V (FloatTensor): The values vector
    """
    # 1. Calculate scores and normalize
    scores = torch.matmul(Q, K.T)
    normalized_scores = scores / math.sqrt(K.shape[0])

    # 2. Apply softmax to get probabilities
    probabilities = torch.softmax(normalized_scores, dim=-1)
    output = torch.matmul(probabilities, V)

    return output


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention block used in the encoder and decoder part of the transformer.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()

        # Assert dimensions
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model / n_heads

        # Query, key and value weights
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
