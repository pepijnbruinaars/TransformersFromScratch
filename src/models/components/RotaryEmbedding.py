import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position information into Q and K vectors via rotation,
    giving the attention mechanism relative position awareness without
    adding absolute positional tokens to the embeddings.

    Applied per attention head on d_k-dimensional vectors, after splitting
    the full d_model into heads but before computing attention scores.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021)

    Args:
        d_k: Head dimension (d_model // n_heads)
        max_seq_len: Maximum sequence length to precompute sin/cos tables for
    """

    def __init__(self, d_k: int, max_seq_len: int) -> None:
        super().__init__()
        # θᵢ = 10000^(-2i / d_k)  for i = 0, 1, ..., d_k/2 - 1
        theta = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2, dtype=torch.float) / d_k))
        # Frequencies per position: (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, theta)
        # Duplicate to cover the full d_k with the split-half rotation: (max_seq_len, d_k)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap and negate the two halves of the last dimension.

        [x₁ ... x_{d/2} | x_{d/2+1} ... x_d]  →  [-x_{d/2+1} ... -x_d | x₁ ... x_{d/2}]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, n_heads, seq_len, d_k)
            k: (batch, n_heads, seq_len, d_k)

        Returns:
            q_rot, k_rot: Rotated query and key, same shapes as input.
        """
        seq_len = q.shape[2]
        # Slice to actual seq_len and broadcast over batch and heads
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot
