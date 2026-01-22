import torch
import torch.nn as nn


class RMSNormalization(nn.Module):
    """RMSNormalization - Just LayerNorm without centering"""

    def __init__(self, epsilon: float = 10**-6):
        super(RMSNormalization, self).__init__()
        self.epsilon = epsilon

        # Multiplicative parameter
        self.alpha = nn.Parameter(torch.ones(1))

        # Additive parameter
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.FloatTensor):
        std = x.std(dim=-1, keepdim=True)

        x_hat = x / torch.sqrt(std + self.epsilon)

        return self.alpha * x_hat + self.bias
