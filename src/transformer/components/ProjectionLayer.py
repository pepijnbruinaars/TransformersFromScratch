import torch.nn as nn


class ProjectionLayer(nn.Module):
    """Layer that projects the output embedding back to the vocabulary."""

    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Return raw logits - CrossEntropyLoss applies softmax internally
        return self.projection_layer(x)
