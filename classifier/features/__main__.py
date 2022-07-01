import logging

import torch

from classifier import Runner
from .model import Model
from .pipeline import Pipeline
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        super().__init__()

        logging.info("\n[--- FEATURE PIPELINE ---]")
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

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self, model=None, collation_fn=None):

        model = Model(
            len(self.pipeline.col_names),
            len(self.data['train'].get_label_keys()),
            config=self.config['model']
        )

        super().__call__(model, self.__collation_fn)

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        label: list = []
        features: list = []

        # collate data
        for sample, review, sentiment in batch:
            label.append(sentiment)
            features.append(
                torch.tensor(
                    sample[self.pipeline.col_names].values,
                    device=get_device()
                )
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
