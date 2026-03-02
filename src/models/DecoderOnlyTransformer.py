import numpy as np
import torch.nn as nn
import logging

from .components import (
    InputEmbedding,
    PositionalEncoding,
    ProjectionLayer,
)
from .components.DecoderOnly.DecoderOnlyStack import DecoderOnlyStack

logger = logging.getLogger(__name__)

def _initialize_weights(module: nn.Module, n_blocks: int) -> None:
    for name, p in module.named_parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
            # Scale output projections by 1/sqrt(2*n_blocks) to prevent
            # residual path explosion (GPT-2 style initialization)
            if "W_O" in name or "linear_2" in name or "down" in name:
                p.data.copy_(p.data / np.sqrt(2 * n_blocks))
        else:
            if "normalization.alpha" in name:
                nn.init.ones_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 dropout: float,
                 vocab_size: int,
                 sequence_length: int,
                 use_flash_attention: bool = True,
                 activation: str = "gelu",
                 use_rope: bool = False):

        super(DecoderOnlyTransformer, self).__init__()
        self.decoder_stack = DecoderOnlyStack(n_blocks,
                                              d_model,
                                              d_ff,
                                              n_heads,
                                              dropout,
                                              use_flash_attention,
                                              activation,
                                              use_rope=use_rope,
                                              sequence_length=sequence_length,
                                            )
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        # RoPE encodes position inside attention, so absolute PE is not needed
        self.positional_encoding = (
            None if use_rope
            else PositionalEncoding(d_model, sequence_length, dropout)
        )
        self.projection_layer = ProjectionLayer(d_model, vocab_size)
        
        # Share embedding and projectoin weights
        self.projection_layer.weight = self.input_embedding.embedding.weight
        
        # Initialize weights
        _initialize_weights(self, n_blocks)
        logger.info("Initialized the transformer model with the following parameters:")
        logger.info(
            f"n_blocks: {n_blocks}, d_model: {d_model}, d_ff: {d_ff}, n_heads: {n_heads}, dropout: {dropout}, sequence_length: {sequence_length}, vocab_size: {vocab_size}, use_flash_attention: {use_flash_attention}, activation: {activation}"
        )

    def forward(self, x, mask=None, return_attentions=False):
        x = self.input_embedding(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        if return_attentions:
            x, attentions = self.decoder_stack(x, mask=mask, return_attentions=True)
            return self.projection_layer(x), attentions
        x = self.decoder_stack(x, mask=mask, return_attentions=False)
        return self.projection_layer(x)