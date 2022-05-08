from typing import List

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from classifier.lib.neural import Model as AbsModel
from classifier.transformer.util import unpad


class Model(AbsModel, nn.Module):

    #
    #
    #  -------- init -----------
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

    #
    #
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
    def forward(self, embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        # pack sentence vectors as a packed sequence
        packed_embeds = rnn.pack_sequence(
            embeds, enforce_sorted=False
        )

        # convert packed representation to a padded representation
        padded_embeds, pad_mask = rnn.pad_packed_sequence(
            packed_embeds, batch_first=True
        )

        # apply MLP to padded sequence
        padded_mlp_out = self.net(padded_embeds)

        return unpad(padded_mlp_out, pad_mask)
