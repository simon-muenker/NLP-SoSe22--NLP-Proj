import logging
from typing import Tuple

import torch

from classifier.base.model import Model as Base
from classifier.features.model import Model as Features
from .._neural import ModelFrame, Perceptron


class Model(ModelFrame):

    #  -------- init -----------
    #
    def __init__(self, in_size: Tuple[int], out_size: int, config: dict):
        super().__init__(in_size, out_size, config)

        self.base = Base(in_size[0], out_size, config=config['base'])
        self.features = Features(in_size[1], out_size, config=config['features'])
        self.output = Perceptron(out_size * 2, out_size)

        logging.info(
            f'> Init Neural Assemble (Base+Features), trainable parameters: '
            f'{len(self) - (len(self.base) + len(self.features))}'
        )

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {

        }

    #  -------- forward -----------
    #
    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.output(
            torch.concat([
                self.base(data[0]), self.features(data[1])
            ], dim=1)
        )
