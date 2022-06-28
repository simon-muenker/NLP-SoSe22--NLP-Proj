import logging
from typing import Tuple, List

import torch
import torch.nn as nn

from .._neural import ModelFrame
from .._neural.util import get_device


class Model(ModelFrame):

    #  -------- __init__ -----------
    #
    def __init__(self, in_size: int, out_size: int, _: dict = None):
        super().__init__(in_size, out_size, {})

        self.net = nn.Linear(in_size, out_size, bias=False).to(get_device())

        logging.info(f'> Init Neural Feature Weight, trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {}

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> torch.Tensor:
        return self.net(data)
