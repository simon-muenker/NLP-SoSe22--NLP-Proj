import logging
from typing import List, Tuple

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
            self.config["in_size"][1] + self.config["out_size"],
            self.config.copy()
        ).to(get_device())

        self.biaffine_w1 = nn.Parameter(torch.ones(self.config["in_size"][1]), requires_grad=True).to(get_device())
        self.biaffine_w2 = nn.Parameter(torch.ones(self.config["in_size"][1]), requires_grad=True).to(get_device())
        self.biaffine_bias = nn.Parameter(torch.zeros(self.config["in_size"][1]), requires_grad=True).to(get_device())

        self.output = nn.Linear(
            self.config["in_size"][1] + self.config["out_size"],
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
    def forward(self, data: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> List[torch.Tensor]:
        embed: torch.Tensor = torch.stack(self.embeds(data[0]))

        return [
            t for t in self.output(
                torch.cat([
                    (embed[:, :-self.config["out_size"]] * self.biaffine_w1)
                    .add(torch.stack(data[1]) * self.biaffine_w2)
                    .add(self.biaffine_bias)
                    .float()
                    ,
                    embed[:, -self.config["out_size"]:]
                ], dim=1)
            )
        ]
