import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.GELU = nn.GELU()

    def forward(self, x):
        x = self.linear_2(self.dropout(self.GELU(self.linear_1(x))))
        return x
