import logging
from typing import Tuple

import torch
import torch.nn as nn

from classifier.lib.neural import Model as AbsModel
from classifier.lib.neural.util import get_device
from classifier.transformer import Model as BERTHead


class Model(AbsModel, nn.Module):

    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.embeds = BERTHead(
                self.config["in_size"][0],
                self.config["in_size"][1],
                self.config.copy()
            ).to(get_device())

        self.dropout = nn.Dropout(0.0)

        self.output = nn.Linear(
            self.config["in_size"][1],
            self.config["out_size"]
        ).to(get_device())

        logging.info(f'> Init Neural Assemble (MLP), trainable parameters: {len(self)- len(self.embeds)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "embeds": BERTHead.default_config(),
            "linguistic": None,
        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.output(self.dropout(self.embeds(data[0]) * data[1]))
