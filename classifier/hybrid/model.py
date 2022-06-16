import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from classifier.lib.neural import Biaffine
from classifier.lib.neural import Model as AbsModel
from classifier.lib.neural.util import get_device
from classifier.transformer import Model as BERTHead


class Model(AbsModel, nn.Module):

    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.embeds = nn.Sequential(
            BERTHead(
                self.config["in_size"][0],
                self.config["in_size"][1] + self.config["out_size"],
                self.config.copy()
            ),
            nn.Dropout(
                p=self.config["dropout"]
            ),
            nn.LeakyReLU()
        ).to(get_device())

        self.biaffine = Biaffine(self.config["in_size"][1], self.config["dropout"])

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
        embed: torch.Tensor = self.embeds(data[0])

        return self.output(
                torch.cat([
                    self.biaffine(
                        embed[:, :-self.config["out_size"]],
                        torch.stack(data[1])
                    ),
                    embed[:, -self.config["out_size"]:]
                ], dim=1)
            )
