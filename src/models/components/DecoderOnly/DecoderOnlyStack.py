import torch.nn as nn

from ...components.FeedForward import FeedForward
from ...components.LayerNormalization import LayerNormalization
from ...components.MultiHeadAttention import MultiHeadAttention

from .DecoderBlock import DecoderBlock

class DecoderOnlyStack(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, use_flash_attention):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self_attention=MultiHeadAttention(d_model, n_heads, dropout, use_flash_attention),
                    feed_forward=FeedForward(d_model, d_ff, dropout),
                    dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.normalization_layer = LayerNormalization(d_model)

    def forward(self, x, mask=None, return_attentions=False):
        all_attentions = []
        for block in self.blocks:
            if return_attentions:
                x, self_attn = block(x, mask, return_attentions=True)
                all_attentions.append(self_attn)
            else:
                x = block(x, mask, return_attentions=False)
        x = self.normalization_layer(x)
        if return_attentions:
            return x, all_attentions
        return x