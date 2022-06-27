import logging
from typing import List

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from classifier.util import timing

COL_NAMES: list = [
    'blob_polarity', 'blob_subjectivity', 'ent_ratio',
    'pos_ratio-noun', 'pos_ratio-verb', 'pos_ratio-adv',
    'pos_ratio-adj', 'pos_ratio-intj', 'pos_ratio-sym'
]

POS_TAGS: list = [
    'NOUN', 'VERB', 'ADV',
    'ADJ', 'INTJ', 'SYM'
]


class SpacyPipe:
    _ = SpacyTextBlob

    #  -------- __init__ -----------
    #
    def __init__(self, name: str = 'en_core_web_sm', disable: List[str] = None):
        if disable is None:
            disable = []

        self.pipeline = spacy.load(name, disable=disable)
        self.pipeline.add_pipe("spacytextblob")

        logging.info(f'> Init Spacy Pipeline: \'{name}\', with: {self.pipeline.pipe_names}')

    #
    #
    #  -------- apply -----------
    #
    @timing
    def apply(self, data: pd.DataFrame, col: str, label: str = '***'):
        logging.info(f'> Apply Space Pipeline to: {label}')

        data[COL_NAMES] = pd.DataFrame.from_records([
            pd.Series([
                doc._.blob.polarity,
                doc._.blob.subjectivity,
                len(doc.ents) / len(doc),
                *[[token.pos_ for token in doc].count(p) / len(doc) for p in POS_TAGS]
            ]) for doc in self.pipeline.pipe(data[col])
        ])

    #  -------- col_names -----------
    #
    @property
    def col_names(self):
        return COL_NAMES
