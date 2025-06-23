import torch.nn as nn

from transformer.components import Encoder, InputEmbedding, PositionalEncoding


class Transformer(nn.Module):
    """Transformer network architecture."""

    def __init__(
        self, encoder_blocks: int, d_model: int, d_ff: int, n_heads: int, dropout: float
    ) -> None:
        """Initialize the transformer architecture

        Args:
            d_model (int): The embedding dimension
        """
        super(Transformer, self).__init__()
        self.enc_embedding = InputEmbedding(512, 5_000)
        self.positional_encoding = PositionalEncoding(d_model, 10, 0.0)
        self.encoder = Encoder(encoder_blocks, d_model, d_ff, n_heads, dropout)

    def forward(self):
        raise NotImplementedError()
