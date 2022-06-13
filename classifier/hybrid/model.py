import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from classifier.lib.neural import Model as AbsModel
from classifier.lib.neural.util import get_device
from classifier.transformer import Model as Linear


class Model(AbsModel, nn.Module):

    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.embeds = Linear(
            self.config["in_size"][0],
            self.config["in_size"][1],
            self.config.copy()
        ).to(get_device())

        self.output = nn.Linear(
            self.config["in_size"][1] * 2,
            self.config["out_size"],
            bias=False
        ).to(get_device())

        logging.info(f'> Init Neural Assemble (MLP), trainable parameters: {len(self)}')

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "embeds": Linear.default_config(),
            "linguistic": None,
        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> List[torch.Tensor]:
        return [t for t in self.output(torch.cat([
            torch.stack(self.embeds(data[0])),
            torch.stack(data[1])
        ], dim=1).float())]
