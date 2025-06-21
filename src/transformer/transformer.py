import torch.nn as nn

from transformer.components import InputEmbedding, PositionalEncoding


class Transformer(nn.Module):
    """Transformer network architecture."""

    def __init__(self, d_model: int) -> None:
        """Initialize the transformer architecture

        Args:
            d_model (int): The embedding dimension
        """
        super(Transformer, self).__init__()
        self.enc_embedding = InputEmbedding(512, 5_000)
        self.positional_encoding = PositionalEncoding(d_model, 10, 0.0)

    def forward(self):
        raise NotImplementedError()
