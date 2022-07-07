import torch
import torch.nn as nn

from .util import get_device


class Perceptron(nn.Module):

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            in_size: int,
            out_size: int,
            dropout: float = 0.5,
            activation_fn: nn.Module = nn.LeakyReLU()
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_size, out_size),
            activation_fn
        ).to(get_device())

    #  -------- forward -----------
    #
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)
