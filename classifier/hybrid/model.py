from typing import List, Tuple

import torch
import torch.nn as nn

from classifier.lib.neural import Model as AbsModel
from classifier.transformer import Model as Linear


class Model(AbsModel, nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.embeds = Linear(
            self.config["in_size"][0],
            self.config["in_size"][1],
            self.config.copy()
        )

        self.output = nn.Linear(
            self.config["in_size"][1] ** 2,
            self.config["out_size"],
            bias=False
        )

        print(self.embeds)
        print(self.output)

    #
    #
    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "embeds": Linear.default_config(),
            "linguistic": None,
            "output": Linear.default_config(),
        }

    #
    #
    #  -------- forward -----------
    #
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> List[torch.Tensor]:

        embeds: List[torch.Tensor] = self.embeds(data[0])
        classes: torch.Tensor = torch.Tensor(data[1])

        output: List[torch.Tensor] = self.output([embeds, classes])

        return output
