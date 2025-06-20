import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()
