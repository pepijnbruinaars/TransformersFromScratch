import torch
import torch.nn as nn

from .FeedForward import FeedForward
from .MultiHeadAttention import MultiHeadAttention
from .ResidualConnection import ResidualConnection
from .LayerNormalization import LayerNormalization


class DecoderBlock(nn.Module):
    """Decoder block. This contains (in order) 1 masked MHA with skip connection to normalization, one MHA with skip connection to layer normalization and one feedforward with skip connection to a normalization layer.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
        self.residual_connection_3 = ResidualConnection(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ):
        # 1. Calculate self-attention
        x = self.residual_connection_1(
            x, lambda x: self.self_attention(x, x, x, decoder_mask)
        )

        # 2. Calculate cross-attention
        x = self.residual_connection_2(
            x,
            lambda x: self.cross_attention(
                x, encoder_output, encoder_output, encoder_mask
            ),
        )

        # 3. Finally, the feed forward block
        x = self.residual_connection_3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(
        self, n_blocks: int, d_model: int, d_ff: int, n_heads: int, dropout: float
    ):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self_attention=MultiHeadAttention(d_model, n_heads, dropout),
                    cross_attention=MultiHeadAttention(d_model, n_heads, dropout),
                    feed_forward=FeedForward(d_model, d_ff, dropout),
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.normalization_layer = LayerNormalization()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.normalization_layer(x)
