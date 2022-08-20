import logging

import numpy as np
import torch

from classifier import Runner
from .pipeline import Pipeline
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        super().__init__()

        logging.info('\n[--- FEATURE PIPELINE ---]')
        self.pipeline = Pipeline(
            target_label=self.data['train'].target_label,
            target_values=self.data['train'].get_label_keys(),
            config=self.config['model']['features']
        )

        self.pipeline.fit(
            self.data['train'].data,
            log_label=self.data['train'].data_path
        )
        self.pipeline.export(self.config['out_path'])

        for data_label, dataset in self.data.items():
            self.pipeline.apply(dataset.data, label=dataset.data_path)
            dataset.data[[dataset.target_label, *self.pipeline.col_names]].to_csv(
                f'{self.config["out_path"]}features.{data_label}.csv'
            )

    #  -------- __call__ -----------
    #
    def __call__(self, *args) -> None:
        super().__call__(len(self.pipeline.col_names), self.__collation_fn)

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        return (
            self.collate_features(batch),
            self.collate_target_label(batch)
        )

    #
    #
    #  -------- collate_features -----------
    #
    def collate_features(self, batch: list) -> torch.Tensor:
        return torch.stack([
            (
                torch.tensor(
                    sample[self.pipeline.col_names].astype('float64').values,
                    device=get_device()
                )
                .squeeze()
                .float()
            ) for sample in batch
        ])


#
#
#  -------- __main__ -----------
#
if __name__ == '__main__':
    Main()()
