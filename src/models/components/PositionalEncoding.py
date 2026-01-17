import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.length = length
        self.dropout = nn.Dropout(dropout)

        # Create matrix and positions tensor
        positional_encodings = torch.zeros(length, d_model)
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)

        # I am aware this is now in log-space for most transformer models because of numerical stability.
        # I'm keeping the original paper's approach for now
        division_term = torch.pow(
            10000.0, -torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        )

        # Apply sin to positions 0, 2, 4, ...
        positional_encodings[:, 0::2] = torch.sin(position * division_term)
        # Apply cos to positions 1, 3, 5, ...
        positional_encodings[:, 1::2] = torch.cos(position * division_term)

        # Add dimension for batched training
        positional_encodings = positional_encodings.unsqueeze(0)

        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x: torch.FloatTensor):
        pos_encoding = self.positional_encodings[:, : x.shape[1], :]  # type: ignore
        value = x + pos_encoding
        return self.dropout(value)
