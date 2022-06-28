import logging
from dataclasses import dataclass

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


@dataclass
class SpacyPipe:
    _ = SpacyTextBlob

    config: dict = None

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            'model': 'en_core_web_sm',
            'pos_tags': [
                'NOUN', 'VERB', 'ADV',
                'ADJ', 'INTJ', 'SYM'
            ],
            'ents': True,
            'blob': True,
            'disabled': []
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self, ):
        if self.config is None:
            self.config = self.default_config

        self.pipeline = spacy.load(self.config['model'], disable=self.config.get('disable', []))

        if self.config.get('blob'):
            self.pipeline.add_pipe("spacytextblob")

        logging.info(f'> Init Spacy Pipeline: \'{self.config["model"]}\', with: {self.pipeline.pipe_names}')

    #
    #
    #  -------- apply -----------
    #
    def apply(self, data: pd.DataFrame, col: str):

        data[self.col_names] = pd.DataFrame.from_records([
            pd.Series([
                doc._.blob.polarity if self.config.get('blob') else None,
                doc._.blob.subjectivity if self.config.get('blob') else None,
                len(doc.ents) / len(doc) if self.config.get('ents') else None,
                *[
                    [token.pos_ for token in doc]
                    .count(p) / len(doc)
                    for p in self.config.get('pos_tags')
                ]
            ])
            .dropna()
            for doc in self.pipeline.pipe(data[col])
        ])

    #  -------- col_names -----------
    #
    @property
    def col_names(self):
        cols: list = []

        if self.config.get('pos_tags'):
            cols += [f'pos_ratio_{pos}' for pos in self.config.get('pos_tags')]

        if self.config.get('ents'):
            cols += ['ent_ratio']

        if self.config.get('blob'):
            cols += ['blob_polarity', 'blob_subjectivity']

        return cols
