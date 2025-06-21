import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    """LayerNormalization"""

    def __init__(self, epsilon: float = 10**-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

        # Multiplicative parameter
        self.alpha = nn.Parameter(torch.ones(1))

        # Additive parameter
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.FloatTensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x_hat = x - mean / math.sqrt(std + self.epsilon)

        return self.alpha * x_hat + self.bias
