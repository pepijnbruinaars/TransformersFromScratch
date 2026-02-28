import torch.nn as nn

from .Swish import Swish

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": Swish,
}


class FeedForward(nn.Module):
    """Standard feed-forward network: Linear -> Activation -> Dropout -> Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(ACTIVATIONS.keys())}")
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = ACTIVATIONS[activation]()

    def forward(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network: Down(Swish(Gate(x)) * Up(x))

    Uses three linear projections instead of two. For equivalent parameter
    count to a standard FFN with d_ff, use d_ff = 2/3 * original_d_ff.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.swish = Swish()

    def forward(self, x):
        return self.down(self.dropout(self.swish(self.gate(x)) * self.up(x)))


def build_feedforward(d_model: int, d_ff: int, dropout: float, activation: str = "gelu") -> nn.Module:
    """Factory function to create the appropriate feed-forward module.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward intermediate dimension
        dropout: Dropout rate
        activation: One of "relu", "gelu", "swish", or "swiglu"

    Returns:
        A FeedForward or SwiGLUFeedForward module
    """
    if activation == "swiglu":
        return SwiGLUFeedForward(d_model, d_ff, dropout)
    return FeedForward(d_model, d_ff, dropout, activation)
