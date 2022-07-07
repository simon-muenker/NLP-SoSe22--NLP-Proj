import logging
from typing import Tuple, List

import torch

from .._neural import ModelFrame, Perceptron


class Model(ModelFrame):

    #  -------- __init__ -----------
    #
    def __init__(self, in_size: int, out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.net = Perceptron(
            in_size, out_size,
            dropout=config["dropout"]
        )

        logging.info(f'> Init BERT-Head (Base), trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "dropout": 0.5
        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> torch.Tensor:
        return self.net(data)
