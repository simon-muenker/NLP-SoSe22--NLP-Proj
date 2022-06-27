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

        self.bertPred = BERTHead(
            self.config["in_size"][0],
            self.config["out_size"],
            self.config.copy()
        ).to(get_device())

        self.bertFeat = BERTHead(
            self.config["in_size"][0],
            self.config["in_size"][1],
            self.config.copy()
        ).to(get_device())

        self.drop = nn.Dropout(self.config["dropout"]).to(get_device())

        self.output = nn.Linear(
            self.config["in_size"][1] + self.config["out_size"],
            self.config["out_size"]
        ).to(get_device())

        logging.info(f'> Init Neural Assemble (MLP), trainable parameters: {len(self) - len(self.bertPred) - len(self.bertFeat)}')

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
        emb: torch.Tensor = self.bertFeat(data[0])

        return self.output(
            self.drop(
                torch.cat([
                    self.bertFeat(data[0]) * data[1],
                    self.bertPred(data[0])
                ], dim=1)
            )
        )
