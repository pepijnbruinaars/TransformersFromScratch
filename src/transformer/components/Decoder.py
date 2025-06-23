import torch.nn as nn


class Decoder(nn.Module):
    """Decoder block. This contains (in order) 1 masked MHA with skip connection to normalization, one MHA with skip connection to layer normalization and one feedforward with skip connection to a normalization layer.

    Args:
        nn (_type_): _description_
    """

    def __init__(self) -> None:
        super(Decoder, self).__init__()
