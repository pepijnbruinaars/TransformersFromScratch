import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """LayerNormalization with per-feature learned parameters."""

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

        # Per-feature learnable parameters (must be d_model-sized, not scalar)
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.FloatTensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x_hat = (x - mean) / (std + self.epsilon)

        return self.alpha * x_hat + self.bias
