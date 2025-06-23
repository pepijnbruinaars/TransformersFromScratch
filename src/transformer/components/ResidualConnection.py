import torch
import torch.nn as nn

from .LayerNormalization import LayerNormalization


class ResidualConnection(nn.Module):
    """"""

    def __init__(self, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: nn.Module):

        return x + self.dropout(sublayer(self.normalization(x)))
