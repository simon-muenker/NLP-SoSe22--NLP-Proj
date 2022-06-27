import logging
from typing import Tuple

import torch
import torch.nn as nn

from .._neural import ModelFrame, MLP
from .._neural.util import get_device


class Model(ModelFrame):

    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.bertPred = MLP(
            self.config["in_size"][0],
            self.config['hid_size'],
            self.config["out_size"],
            dropout=self.config['dropout']
        )

        self.bertFeat = MLP(
            self.config["in_size"][0],
            self.config['hid_size'],
            self.config["in_size"][1],
            dropout=self.config['dropout']
        )

        self.drop = nn.Dropout(self.config["dropout"]).to(get_device())

        self.output = nn.Linear(
            self.config["in_size"][1] + self.config["out_size"],
            self.config["out_size"]
        ).to(get_device())

        logging.info(f'> Init Neural Assemble (MLP), trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {

        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.output(
            self.drop(
                torch.cat([
                    self.bertFeat(data[0]) * data[1],
                    self.bertPred(data[0])
                ], dim=1)
            )
        )
