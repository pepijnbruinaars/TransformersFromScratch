import torch
import torch.nn as nn

from .components import (
    Encoder,
    Decoder,
    InputEmbedding,
    PositionalEncoding,
    ProjectionLayer,
)


def _initialize_weights(module: nn.Module) -> None:
    """Private function to initialize the weights of the transformer model using Xavier initialization.

    Args:
        module (nn.Module): The module to initialize.
    """
    for name, p in module.named_parameters():
        # We only want to initialize weights
        if p.dim() > 1:
            nn.init.normal_(p)


class Transformer(nn.Module):
    """Transformer network architecture."""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        source_length: int,
        target_length: int,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
    ) -> None:
        """Create a new Transformer model.

        This model consists of an encoder and a decoder, each consisting of `n_blocks` blocks. The encoder processes the input sequence, while the decoder generates the output sequence based on the encoded input.

        Args:
            n_blocks (int): The number of encoder/decoder blocks in the encoder/decoder
            d_model (int): The dimensionality of the embedding space in the model. Can be thought of as the "internal resolution" of the model.
            d_ff (int): The dimensionality of the feed-forward network in the model. Typically, `d_ff` > `d_model` to allow for complex transformations.
            n_heads (int): The number of attention heads in the multi-head attention mechanism. This allows the model to focus on different parts of the input sequence simultaneously. Must be a divisor of `d_model`.
            dropout (float): Probability of dropout to apply to the model. Is used throughout the model to prevent overfitting.
            source_length (int): The maximum length of the input sequence that the model can handle. This is used to create positional encodings and input embeddings.
            target_length (int): The maximum length of the output sequence that the model can handle. This is used to create positional encodings and input embeddings.
            source_vocabulary_size (int): The size of the vocabulary for the input sequence. This is used to create the input embedding layer.
            target_vocabulary_size (int): The size of the vocabulary for the output sequence. This is used to create the output embedding layer and the projection layer.
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = InputEmbedding(d_model, source_vocabulary_size)
        self.decoder_embedding = InputEmbedding(d_model, target_vocabulary_size)
        self.encoder_positional_encoding = PositionalEncoding(
            d_model, source_length, dropout
        )
        self.decoder_positional_encoding = PositionalEncoding(
            d_model, target_length, dropout
        )
        self.encoder = Encoder(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.projection_layer = ProjectionLayer(d_model, target_vocabulary_size)

        self.projection_layer.weight = self.decoder_embedding.embedding.weight

        # Initialize parameters using He initialization
        _initialize_weights(self)
        print("Initialized the transformer model with the following parameters:")
        print(
            f"n_blocks: {n_blocks}, d_model: {d_model}, d_ff: {d_ff}, n_heads: {n_heads}, dropout: {dropout}, source_length: {source_length}, target_length: {target_length}, source_vocabulary_size: {source_vocabulary_size}, target_vocabulary_size: {target_vocabulary_size}"
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the input sequence."""
        x = self.encoder_embedding(x)
        x = self.encoder_positional_encoding(x)
        return self.encoder(x, mask)

    def decode(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the input sequence."""
        x = self.decoder_embedding(x)
        x = self.decoder_positional_encoding(x)
        return self.decoder(x, encoder_output, encoder_mask, decoder_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project the output to the vocabulary size."""
        normalized_x = self.decoder.normalization_layer(x)
        return self.projection_layer(normalized_x)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run full forward pass: encode -> decode -> project.

        Returns projection logits of shape (batch, seq_len, target_vocabulary_size).
        """
        encoder_output = self.encode(source, source_mask)
        decoder_output = self.decode(target, encoder_output, source_mask, target_mask)
        return self.project(decoder_output)
