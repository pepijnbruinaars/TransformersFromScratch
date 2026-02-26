# Decoder block only has a decoder - so self-attention and feedforward. No cross attention.
import torch
import torch.nn as nn

from ...components.ResidualConnection import ResidualConnection

class DecoderBlock(nn.Module):
    """A single decoder block consisting of self-attention and feedforward layers, with residual connections."""
    
    def __init__(
        self,
        self_attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection_1 = ResidualConnection(dropout)
        self.residual_connection_2 = ResidualConnection(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        decoder_mask: torch.Tensor,
        return_attentions: bool = False,
    ):
        self_attn = None

        # 1. Calculate self-attention
        out1 = self.residual_connection_1(
            x, lambda x: self.self_attention(x, x, x, decoder_mask, return_attentions=return_attentions)
        )
        if return_attentions and isinstance(out1, tuple):
            x, self_attn = out1
        else:
            x = out1

        # 2. Finally, the feed forward block
        out2 = self.residual_connection_2(x, self.feed_forward)
        if isinstance(out2, tuple):
            x = out2[0]

        if return_attentions:
            return x, self_attn
        return x