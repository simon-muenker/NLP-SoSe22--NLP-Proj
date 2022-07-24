import logging
from abc import abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn

from classifier.util import dict_merge, byte_to_mb
from .util import get_device, memory_usage


class Model(nn.Module):

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "name": "Dropout->Dense->ELU",
            "in_size": 64,
            "out_size": 2,
            "dropout": 0.0,
        }

    #  -------- init -----------
    #
    def __init__(self, user_config: dict = None):
        super().__init__()
        self.config: dict = Model.default_config()
        dict_merge(self.config, user_config)

        self.net = nn.Sequential(
            nn.Dropout(p=self.config["dropout"]),
            nn.Linear(
                self.config["in_size"],
                self.config["in_size"]
            ),
            nn.ELU(),
            nn.Linear(
                self.config["in_size"],
                self.config["out_size"]
            ),
            nn.ELU()
        ).to(get_device())

        logging.info((
            f'> Init {self.config["name"]}\n'
            f'  Memory Usage: {byte_to_mb(memory_usage(self))}\n'
            f'  Trainable parameters: {len(self)}\n'
            f'  Input Dimension: {self.config["in_size"]}\n'
            f'  Output Dimension: {self.config["out_size"]}'
        ))

    #  -------- forward -----------
    #
    @abstractmethod
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> torch.Tensor:
        return self.net(data)

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
        pred_label = self(embeds)

        # compute loss, backward
        return (
            loss_fn(pred_label, gold_label),
            torch.argmax(pred_label, dim=1)
        )

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
        save_state = torch.load(path, map_location=get_device())

        model: nn.Module = cls(save_state["config"]).to(get_device())
        model.load_state_dict(save_state["state_dict"])

        return model

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
