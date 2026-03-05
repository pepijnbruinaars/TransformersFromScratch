from typing import Optional
import torch.nn as nn

from ...components.FeedForward import build_feedforward
from ...components.LayerNormalization import LayerNormalization
from ...components.MultiHeadAttention import MultiHeadAttention

from .DecoderBlock import DecoderBlock
from .KVCache import KVCache


class DecoderOnlyStack(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, use_flash_attention, activation="gelu", use_rope=False, sequence_length=512):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    self_attention=MultiHeadAttention(d_model, n_heads, dropout, use_flash_attention, use_rope=use_rope, sequence_length=sequence_length),
                    feed_forward=build_feedforward(d_model, d_ff, dropout, activation),
                    dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.normalization_layer = LayerNormalization(d_model)

    def forward(self, x, mask=None, return_attentions=False, cache: Optional[KVCache] = None):
        all_attentions = []
        for i, block in enumerate(self.blocks):
            if return_attentions:
                x, self_attn = block(x, mask, cache=cache, layer_idx=i, return_attentions=True)
                all_attentions.append(self_attn)
            else:
                x = block(x, mask, cache=cache, layer_idx=i, return_attentions=False)
        x = self.normalization_layer(x)
        if return_attentions:
            return x, all_attentions
        return x