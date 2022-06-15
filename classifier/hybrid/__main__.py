import logging

import torch

from classifier.hybrid import Model as Network
from classifier.hybrid import SpacyPipe
from classifier.lib import Runner
from classifier.lib.neural import Encoding, Trainer
from classifier.lib.neural.util import get_device
from classifier.linguistic import Model as Classifier

DROP_COLS: list = [
    '1-gram', '2-gram',
    'sum_negative', 'sum_positive',
    'prediction'
]


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # --- ---------------------------------
        # --- load components

        # load encoding, classifier, spacy
        self.encoding = Encoding(self.config['model']['encoding'])
        self.clss = Classifier(self.config['model']['linguistic'])
        self.spacy = SpacyPipe()

        # load network
        self.net = Network(
            in_size=tuple([
                self.encoding.dim,
                len(self.clss.col_names) + len(self.spacy.col_names)
            ]),
            out_size=len(self.data['train'].get_label_keys()),
            config=self.config['model']['neural']
        )

        # load trainer
        self.trainer = Trainer(
            self.net,
            self.data,
            self.collation_fn,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

    #  -------- __call__ -----------
    #
    def __call__(self):
        logging.info("\n[--- RUN ---]")

        # --- ---------------------------------
        # --- fit, save
        self.clss.fit(self.data['train'].data, label=self.data['train'].data_path)
        self.clss.save(self.config['out_path'])

        for data_label, dataset in self.data.items():
            if data_label not in self.config["data"]["eval_on"]:
                continue

            # predict classifier
            self.clss.predict(dataset.data, label=dataset.data_path)

            # drop legacy columns
            dataset.data.drop(columns=DROP_COLS, inplace=True)

            # apply spacy pipeline
            self.spacy.apply(dataset.data, 'review', label=dataset.data_path)

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
        clss_pred: list = []

        # collate data
        for sample, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)
            clss_pred.append(torch.tensor(sample[[
                *self.clss.col_names,
                *self.spacy.col_names
            ]].values, device=get_device()).squeeze())

        # embed text
        _, sent_embeds, _ = self.encoding(text)

        # extract only first embeddings (CLS)
        cls_embeds: list = [tco[0] for tco in sent_embeds]

        # transform labels
        label_ids: torch.Tensor = torch.tensor(
            [self.data['train'].encode_label(lb) for lb in label],
            dtype=torch.long, device=get_device()
        )

        return (cls_embeds, clss_pred), label_ids


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
