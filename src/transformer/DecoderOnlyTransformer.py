import torch
import torch.nn as nn

from transformer.components import InputEmbedding, PositionalEncoding, Decoder, ProjectionLayer

def _initialize_weights(model: nn.Module) -> None:
    """Initialize the weights of the model using Xavier initialization."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class DecoderOnlyTransformer(nn.Module):
    """Some Information about DecoderOnlyTransformer"""

    def __init__(self,
                 n_blocks: int,
                 d_model: int,
                 d_ff: int,
                 n_heads: int,
                 dropout: float,
                 sequence_length: int,
                 vocabulary_size: int) -> None:
        """Create a new DecoderOnlyTransformer model.
        
        n_blocks (int): The number of encoder/decoder blocks in the encoder/decoder
        d_model (int): The dimensionality of the embedding space in the model. Can be thought of as the "internal resolution" of the model.
        d_ff (int): The dimensionality of the feed-forward network in the model. Typically, `d_ff` > `d_model` to allow for complex transformations.
        n_heads (int): The number of attention heads in the multi-head attention mechanism. This allows the model to focus on different parts of the input sequence simultaneously. Must be a divisor of `d_model`.
        dropout (float): Probability of dropout to apply to the model. Is used throughout the model to prevent overfitting.
        sequence_length (int): The maximum length of the sequence that the model can handle. This is used to create positional encodings and input embeddings.
        vocabulary_size (int): The size of the vocabulary for the sequence. This is used to create the input embedding layer.
        """
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = InputEmbedding(d_model, vocabulary_size)
        self.positional_encoding = PositionalEncoding(d_model, sequence_length, dropout)
        self.decoder = Decoder(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.projection_layer = ProjectionLayer(d_model, vocabulary_size)

        _initialize_weights(self)
        print("Initialized the decoder-only transformer model with the following parameters:")
        print(
            f"n_blocks: {n_blocks}, d_model: {d_model}, d_ff: {d_ff}, n_heads: {n_heads}, dropout: {dropout}, sequence_length: {sequence_length}, vocabulary_size: {vocabulary_size}"
        )

    def forward(self, x):
        raise NotImplementedError()
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project the output of the decoder to the vocabulary size using the projection layer.

        Args:
            x (torch.Tensor): The output of the decoder.

        Returns:
            torch.Tensor: The projected output.
        """
        return self.projection_layer(x)
