import logging
from typing import List

import torch
import torch.nn as nn

from classifier.lib.neural import Model as AbsModel


class Model(AbsModel, nn.Module):

    #  -------- __init__ -----------
    #
    def __init__(self, in_size: int, out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.net = nn.Sequential(
            nn.Linear(
                self.config["in_size"],
                self.config["hid_size"]),
            nn.Dropout(
                p=self.config["dropout"]),
            nn.LeakyReLU(),
            nn.Linear(
                self.config["hid_size"],
                self.config["out_size"])
        )

        logging.info(f'> Init BERT-Head (MLP), trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "hid_size": 64,
            "dropout": 0.2
        }

    #
    #
    #  -------- forward -----------
    #
    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.net(embeds)
