import logging
from typing import Dict

import pandas as pd

from classifier.util import timing
from .groupcounter import GroupCounter
from .spacy import SpacyPipe


class Pipeline:

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            target_values: list,
            config: dict
    ) -> None:

        self.target_values = target_values
        self.config = config

        if self.config.get("ngram_counter", None):
            self.ngram_counter: Dict[str, GroupCounter] = {}

        if self.config.get("spacy_pipeline", None):
            self.spacy = SpacyPipe()

        logging.info(f'> Init N-Gram Group Counter, with: {list(self.config["ngram_counter"].items())}')

    #  -------- fit -----------
    #
    @timing
    def fit(self, data: pd.DataFrame, target_label: str, log_label: str = '***') -> None:

        if self.config.get("ngram_counter", None):
            logging.info(f'> Fit N-Gram Group Counter on {log_label}')
            for n, keep in self.config['ngram_counter'].items():
                self.ngram_counter[n] = GroupCounter(
                    data,
                    key_label=f'{n}-gram',
                    group_label=target_label,
                    keep=keep
                )

    #
    #
    #  -------- predict -----------
    #
    @timing
    def apply(self, data: pd.DataFrame, label: str = '***') -> None:
        logging.info(f'> Predict with Freq. Classifier on {label}')

        # calculate the ngram_counter scores
        if self.config.get("ngram_counter", None):
            for n, gc in self.ngram_counter.items():
                gc.predict(data)

        # apply spacy pipeline
        if self.config.get("spacy_pipeline", None):
            self.spacy.apply(data, 'review', label=label)

    #  -------- export -----------
    #
    def export(self, path: str):
        if self.config.get("ngram_counter", None):
            for n, gc in self.ngram_counter.items():
                gc.write(f'{path}{n}-gram-weights')

    #  -------- property -----------
    #
    @property
    def col_names(self):
        cols: list = []

        if self.config.get("ngram_counter", None):
            cols += sum([pc.col_names for pc in self.ngram_counter.values()], [])

        if self.config.get("spacy_pipeline", None):
            cols += self.spacy.col_names

        return cols
