from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from .util import unpad, get_device


class Model(nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, in_size: int, out_size: int, config: dict):
        super().__init__()

        if config is None:
            config = self.default_config()

        self.config = config
        self.config["in_size"] = in_size
        self.config["out_size"] = out_size

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

    #
    #
    #  -------- train_step -----------
    #
    def train_step(
            self,
            loss_fn: torch.nn.Module,
            batch: Tuple[torch.Tensor, List[torch.Tensor]]):
        # predict batch
        embeds, gold_label = batch
        pred_label = torch.stack(self(embeds))

        # compute loss, backward
        return (
            loss_fn(pred_label, gold_label),
            torch.argmax(pred_label, dim=1)
        )

    #
    #
    #  -------- predict -----------
    #
    @torch.no_grad()
    def predict(self, embeds: List[torch.Tensor]) -> List[int]:
        return list(torch.argmax(self(embeds), dim=1))

    #  -------- save -----------
    #
    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict()
            },
            path,
        )

    #  -------- load -----------
    #
    @classmethod
    def load(cls, path: str) -> nn.Module:
        data = torch.load(path, map_location=get_device())

        model: nn.Module = cls(
            data["config"]["in_size"],
            data["config"]["out_size"],
            data["config"]
        ).to(get_device())
        model.load_state_dict(data["state_dict"])

        return model

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
