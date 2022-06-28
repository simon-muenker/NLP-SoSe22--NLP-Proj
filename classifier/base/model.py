import logging
from typing import Tuple, List

import torch

from .._neural import ModelFrame, MLP


class Model(ModelFrame):

    #  -------- __init__ -----------
    #
    def __init__(self, in_size: int, out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.net = MLP(
            self.config["in_size"],
            self.config["hid_size"],
            self.config["out_size"],
            dropout=self.config["dropout"]
        )

        logging.info(f'> Init BERT-Head (Base), trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "hid_size": 64,
            "dropout": 0.2
        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> torch.Tensor:
        return self.net(data)
