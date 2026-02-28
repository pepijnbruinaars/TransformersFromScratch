import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function: Swish(x) = x * sigmoid(x)"""

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)
