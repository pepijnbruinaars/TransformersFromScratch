import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    """The input embedding wrapper"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """_summary_

        Args:
            d_model (int): The embedding dimensions
            vocab_size (int): The size of the vocabulary
        """
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: int):
        return self.embedding(x) * math.sqrt(self.d_model)
