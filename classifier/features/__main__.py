import logging

import torch

from classifier import Runner, Metric
from .model import Model
from .pipeline import Pipeline
from .._neural import Trainer
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        super().__init__()
        self.metric = Metric()
        self.pipeline = Pipeline(self.config['model'])

        self.model = Model(
            len(self.pipeline.col_names),
            len(self.data['train'].get_label_keys())
        )

        # trainer
        self.trainer = Trainer(
            self.model,
            self.data,
            self.collation_fn,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):
        logging.info("\n[--- RUN ---]")

        self.pipeline.fit(self.data['train'].data, label=self.data['train'].data_path)
        self.pipeline.save(self.config['out_path'])

        # predict train, eval
        logging.info(f'\n[--- EVAL -> {self.config["data"]["eval_on"]} ---]')
        for data_label, dataset in self.data.items():
            if data_label not in self.config["data"]["eval_on"]:
                continue

            # predict dataset
            self.pipeline.predict(dataset.data, label=dataset.data_path)

        self.trainer()

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        label: list = []
        features: list = []

        # collate data
        for sample, review, sentiment in batch:
            label.append(sentiment)
            features.append(
                torch.tensor(sample[[
                    *self.pipeline.col_names
                ]].values, device=get_device())
                .squeeze()
                .float()
            )

        return (
            torch.stack(features),
            torch.tensor(
                [self.data['train'].encode_label(lb) for lb in label],
                dtype=torch.long, device=get_device()
            )
        )


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
