import logging
from typing import List

import spacy
import torch

from classifier.hybrid import Model as Network
from classifier.hybrid.spacy_util import PretokenizedTokenizer
from classifier.lib import Runner
from classifier.lib.neural import Encoding, Trainer
from classifier.lib.neural.util import get_device
from classifier.linguistic import Model as Classifier


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
            in_size=tuple([self.encoding.dim, 4]),
            out_size=2,
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
        # --- fit, save, predict classifier
        self.clss.fit(self.data['train'].data, label=self.data['train'].data_path)
        self.clss.save(self.config['out_path'])

        self.clss.predict(self.data['train'].data, label=self.data['train'].data_path)
        self.clss.predict(self.data['eval'].data, label=self.data['eval'].data_path)

        # TODO apply
        # self.apply_pipeline(-- some data: pd.DF --)

        # --- ---------------------------------
        # --- train
        self.trainer()

    #
    #  TODO IMPLEMENT
    #  -------- apply_pipeline -----------
    #
    def apply_pipeline(self, data):
        def sc(row: str) -> float:
            doc = self.spacy(row)

            return len(doc.ents) / len(doc)

        data['ent_ratio'] = data["1-gram"].parallel_apply(sc)

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []
        clss_pred: list = []

        # FIXME COLLATION
        # collate data
        for sample, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)
            clss_pred.append(torch.tensor(sample[[
                '1-gram_negative',
                '1-gram_positive',
                '2-gram_negative',
                '2-gram_positive'
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
        # python -m spacy download en_core_web_sm

        if disable is None:
            disable = ['tok2vec', 'parser', 'attribute_ruler', 'lemmatizer']

        pipeline = spacy.load(name, disable=disable)
        pipeline.tokenizer = PretokenizedTokenizer(pipeline.vocab)
        logging.info(f'> Init Spacy Pipeline: \'{name}\', with: {pipeline.pipe_names}')

        return pipeline


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
