import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from classifier.util import timing
from .groupcounter import GroupCounter
from .spacy import SpacyPipe


@dataclass
class Pipeline:
    target_label: str
    target_values: list
    config: dict

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            'ngram_counter': {
                '1': 256,
                '2': 2048
            },
            'spacy_pipeline': SpacyPipe.default_config
        }

    #  -------- __init__ -----------
    #
    def __post_init__(self) -> None:
        if self.config is None:
            self.config = self.default_config

        if self.config.get("ngram_counter", None):
            self.ngram_counter: Dict[str, GroupCounter] = {}

        if self.config.get("spacy_pipeline", None):
            self.spacy = SpacyPipe()

        logging.info(f'> Init N-Gram Group Counter, with: {list(self.config["ngram_counter"].items())}')

    #  -------- fit -----------
    #
    @timing
    def fit(self, data: pd.DataFrame, log_label: str = '***') -> None:
        logging.info(f'> Fit Pipeline {log_label} on (only N-Gram Group Counter)')

        if self.config.get("ngram_counter", None):
            for n, keep in self.config['ngram_counter'].items():
                self.ngram_counter[n] = GroupCounter(
                    data, key_label=f'{n}-gram',
                    group_label=self.target_label,
                    keep=keep
                )

    #
    #
    #  -------- predict -----------
    #
    @timing
    def apply(self, data: pd.DataFrame, label: str = '***') -> None:
        logging.info(f'> Apply Feature Pipeline on {label}')

        # calculate the ngram_counter scores
        if self.config.get("ngram_counter", None):
            for n, gc in self.ngram_counter.items():
                gc.predict(data)

        # apply spacy pipeline
        if self.config.get("spacy_pipeline", None):
            self.spacy.apply(data, 'review')

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
