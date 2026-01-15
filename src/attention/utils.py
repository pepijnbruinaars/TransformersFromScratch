"""
Attention processing utilities.

Functions for extracting, processing, and analyzing attention patterns
from transformer models.
"""

import torch
from typing import Dict, List, Any


def extract_cross_attentions(attentions_dict: Dict[str, Any]) -> List[torch.Tensor]:
    """Extract cross-attention tensors from a model's attention output.

    Args:
        attentions_dict: Dictionary returned by model with return_attentions=True

    Returns:
        List of cross-attention tensors, one per decoder block
    """
    if attentions_dict is None:
        return []

    cross_atts = attentions_dict.get("cross_attentions", [])
    if not cross_atts:
        return []

    return cross_atts


def extract_self_attentions(attentions_dict: Dict[str, Any]) -> List[torch.Tensor]:
    """Extract self-attention tensors from a model's attention output.

    Args:
        attentions_dict: Dictionary returned by model with return_attentions=True

    Returns:
        List of self-attention tensors, one per decoder block
    """
    if attentions_dict is None:
        return []

    self_atts = attentions_dict.get("self_attentions", [])
    if not self_atts:
        return []

    return self_atts


def aggregate_attention_over_sequence(
    attention_sequence: List[torch.Tensor],
) -> torch.Tensor:
    """Aggregate attention patterns over a sequence of decoding steps.

    Args:
        attention_sequence: List of attention tensors from each decoding step

    Returns:
        Final attention tensor (from last decoding step)
    """
    if not attention_sequence:
        raise ValueError("Attention sequence is empty")

    # Return the final (complete) attention matrix
    return attention_sequence[-1]


def average_attention_heads(attn_tensor: torch.Tensor) -> torch.Tensor:
    """Average attention weights across all heads.

    Args:
        attn_tensor: shape (batch, n_heads, seq_len_q, seq_len_k)

    Returns:
        Averaged attention: shape (batch, seq_len_q, seq_len_k)
    """
    return attn_tensor.mean(dim=1)


def get_attention_at_position(
    attn_tensor: torch.Tensor,
    query_pos: int,
    key_pos: int | None = None,
) -> torch.Tensor:
    """Extract attention weights for a specific query position.

    Args:
        attn_tensor: shape (batch, n_heads, seq_len_q, seq_len_k)
        query_pos: Position in query sequence
        key_pos: If specified, get attention to this key position only

    Returns:
        Attention weights: shape (batch, n_heads) or (batch, n_heads, 1) if key_pos specified
    """
    if key_pos is not None:
        return attn_tensor[:, :, query_pos, key_pos]
    else:
        return attn_tensor[:, :, query_pos, :]  # All key positions


def find_max_attention_positions(
    attn_tensor: torch.Tensor,
    top_k: int = 3,
) -> List[List[tuple[int, float]]]:
    """Find the top-k positions with highest attention for each query position.

    Args:
        attn_tensor: shape (batch, n_heads, seq_len_q, seq_len_k)
        top_k: Number of top positions to return

    Returns:
        List of lists: [query_pos][(key_pos, weight), ...]
    """
    batch, n_heads, seq_len_q, seq_len_k = attn_tensor.shape

    results = []
    for query_pos in range(seq_len_q):
        query_results = []
        for head in range(n_heads):
            # Get attention weights for this head and query position
            weights = attn_tensor[0, head, query_pos, :]  # (seq_len_k,)

            # Find top-k positions
            top_values, top_indices = torch.topk(weights, min(top_k, seq_len_k))

            for val, idx in zip(top_values.tolist(), top_indices.tolist()):
                query_results.append((idx, val))

        # Sort by weight and take top-k across heads
        query_results.sort(key=lambda x: x[1], reverse=True)
        results.append(query_results[:top_k])

    return results
