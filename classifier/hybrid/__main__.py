import logging
from typing import List

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import torch

from classifier.hybrid import Model as Network
from classifier.lib import Runner
from classifier.lib.neural import Encoding, Trainer
from classifier.lib.neural.util import get_device
from classifier.lib.util import timing
from classifier.linguistic import Model as Classifier

COLS: dict = {
    'linguistic': [
        '1-gram_negative', '1-gram_positive',
        '2-gram_negative', '2-gram_positive'
    ],
    'spacy': [
        'blob_polarity', 'blob_subjectivity', 'ent_ratio',
        'pos_ratio-noun', 'pos_ratio-verb', 'pos_ratio-adv',
        'pos_ratio-adj', 'pos_ratio-intj', 'pos_ratio-sym'
    ],
    '_drop': [
        '1-gram', '2-gram', 'sum_negative', 'sum_positive', 'prediction'
    ]
}

POS_TAGS: list = [
    'NOUN', 'VERB', 'ADV',
    'ADJ', 'INTJ', 'SYM'
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
        self.spacy = Main.__load_spacy()

        # load network
        self.net = Network(
            in_size=tuple([
                self.encoding.dim,
                len(COLS['linguistic']) + len(COLS['spacy'])
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
            dataset.data.drop(columns=COLS['_drop'], inplace=True)

            # apply spacy pipeline
            self.apply_spacy(dataset.data, 'review', label=dataset.data_path)

        # --- ---------------------------------
        # --- train
        self.trainer()

    #
    #
    #  -------- apply_spacy -----------
    #
    @timing
    def apply_spacy(self, data: pd.DataFrame, col: str, label: str = '***'):
        logging.info(f'> Apply Space Pipeline to: {label}')

        def sc(row: str) -> pd.Series:
            doc = self.spacy(row)
            pos: list = [token.pos_ for token in doc]

            return pd.Series([
                doc._.blob.polarity,
                doc._.blob.subjectivity,
                len(doc.ents) / len(doc),
                *[pos.count(p) / len(doc) for p in POS_TAGS],
            ])

        data[COLS['spacy']] = data[col].apply(sc)

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
                *COLS['linguistic'],
                *COLS['spacy']
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

    #  -------- __load_spacy -----------
    #
    @staticmethod
    def __load_spacy(name: str = 'en_core_web_sm', disable: List[str] = None):
        if disable is None:
            disable = []

        pipeline = spacy.load(name, disable=disable)
        pipeline.add_pipe("spacytextblob")

        logging.info(f'> Init Spacy Pipeline: \'{name}\', with: {pipeline.pipe_names}')

        return pipeline


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
