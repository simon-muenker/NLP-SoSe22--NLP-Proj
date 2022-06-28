import torch

from classifier import Runner
from classifier.features.pipeline import Pipeline as Pipeline
from .model import Model
from .._neural import Encoder, Trainer
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # --- ---------------------------------
        # --- load components

        # load encoding, classifier
        self.encoder = Encoder(self.config['model']['encoding'])
        self.pipeline = Pipeline(self.config['model']['features'])

        # load model
        self.model = Model(
            in_size=tuple([
                self.encoder.dim,
                len(self.pipeline.col_names)
            ]),
            out_size=len(self.data['train'].get_label_keys()),
            config=self.config['model']['base']
        )

        # load trainer
        self.trainer = Trainer(
            self.model,
            self.data,
            self.collation_fn,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

    #  -------- __call__ -----------
    #
    def __call__(self):

        # fit, export pipeline
        self.pipeline.fit(self.data['train'].data, label=self.data['train'].data_path)
        self.pipeline.export(self.config['out_path'])

        # apply pipeline
        for data_label, dataset in self.data.items():
            self.pipeline.apply(dataset.data, label=dataset.data_path)

        # --- ---------------------------------
        # --- train
        self.trainer()

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []
        pipeline: list = []

        # collate data
        for sample, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)
            pipeline.append(
                torch.tensor(sample[[
                    *self.pipeline.col_names,
                ]].values, device=get_device())
                .squeeze()
                .float()
            )

        # embed text
        _, sent_embeds, _ = self.encoder(text, return_unpad=False)

        return (
            (sent_embeds[:, 1], torch.stack(pipeline)),
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
