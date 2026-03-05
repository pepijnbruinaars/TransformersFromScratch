import torch


class KVCache:
    """Per-layer key-value cache for decoder-only generation.

    Stores K and V tensors (shape: batch, n_heads, seq, d_k) per layer.
    Populated during prefill; extended by one position each decode step.
    """

    def __init__(self, n_layers: int) -> None:
        self._k: list[torch.Tensor | None] = [None] * n_layers
        self._v: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new_k/new_v to cache and return full (K, V) for attention."""
        if self._k[layer_idx] is None:
            self._k[layer_idx] = new_k
            self._v[layer_idx] = new_v
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        return self._k[layer_idx], self._v[layer_idx]

    def get_length(self, layer_idx: int) -> int:
        """Number of positions currently cached for this layer."""
        return 0 if self._k[layer_idx] is None else self._k[layer_idx].shape[2]
