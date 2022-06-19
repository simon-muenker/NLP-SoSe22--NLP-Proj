import torch
import torch.nn as nn

from classifier.lib.neural.util import get_device


class Biaffine(nn.Module):

    #  -------- init -----------
    #
    def __init__(self, size: int, dropout: float = 0.2):
        super(Biaffine).__init__()

        self.w1 = nn.Parameter(torch.ones(size), requires_grad=True).to(get_device())
        self.w2 = nn.Parameter(torch.ones(size), requires_grad=True).to(get_device())
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True).to(get_device())

        self.dropout = nn.Dropout(p=dropout)
        self.acf = nn.LeakyReLU()

    #  -------- forward -----------
    #
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.acf((self.dropout(x1) * self.w1) + (self.dropout(x2) * self.w2) + self.bias)
