import torch
import torch.nn as nn

from ..ResidualConnection import ResidualConnection
from ..FeedForward import FeedForward
from ..MultiHeadAttention import MultiHeadAttention
from ..LayerNormalization import LayerNormalization


class EncoderBlock(nn.Module):
    """Encoder block. This contains (in order) 1 MHA with skip connection to normalization, and one feedforward with skip connection to a normalization layer.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attentions: bool = False):
        encoder_attentions = []

        out1 = self.residual_connection_1(
            x, lambda x: self.attention(x, x, x, mask, return_attentions=return_attentions)
        )
        if return_attentions and isinstance(out1, tuple):
            x, attn1 = out1
            encoder_attentions.append(attn1)
        else:
            x = out1

        out2 = self.residual_connection_2(x, self.feed_forward)
        if isinstance(out2, tuple):
            x = out2[0]

        if return_attentions:
            return x, encoder_attentions
        return x


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(
        self, n_blocks: int, d_model: int, d_ff: int, n_heads: int, dropout: float, use_flash_attention: bool = True
    ):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    attention=MultiHeadAttention(d_model, n_heads, dropout, use_flash_attention),
                    feed_forward=FeedForward(d_model, d_ff, dropout),
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.normalization_layer = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attentions: bool = False):
        all_attentions = []
        for layer in self.layers:
            if return_attentions:
                x, atts = layer(x, mask, return_attentions=return_attentions)
                all_attentions.append(atts)
            else:
                x = layer(x, mask)
        x = self.normalization_layer(x)
        if return_attentions:
            return x, all_attentions
        return x
