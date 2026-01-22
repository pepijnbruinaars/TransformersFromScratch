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
        # Call sublayer with normalized input. The sublayer may return either
        # a tensor or a tuple (tensor, attentions). Support both.
        result = sublayer(self.normalization(x))

        if isinstance(result, tuple):
            output, attentions = result
        else:
            output, attentions = result, None

        res = x + self.dropout(output)
        if attentions is not None:
            return res, attentions
        return res
