from abc import abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .util import get_device


class ModelFrame(nn.Module):

    #  -------- init -----------
    #
    def __init__(self, in_size: Union[Tuple, int], out_size: int, config: dict):
        super().__init__()

        if config is None:
            config = self.default_config()

        self.config = config
        self.config["in_size"] = in_size
        self.config["out_size"] = out_size

        self.to(get_device())

    #  -------- default_config -----------
    #
    @staticmethod
    @abstractmethod
    def default_config() -> dict:
        pass

    #  -------- forward -----------
    #
    @abstractmethod
    def forward(self, data: Tuple[List[torch.Tensor], List]) -> torch.Tensor:
        pass

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
