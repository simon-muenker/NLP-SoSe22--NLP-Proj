import torch
import torch.nn as nn

from .util import get_device


class MLP(nn.Module):

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            in_size: int,
            hid_size: int,
            out_size: int,
            dropout: float = 0.5
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_size, hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, out_size)
        ).to(get_device())

    #  -------- forward -----------
    #
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)
