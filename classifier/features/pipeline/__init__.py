import itertools
import logging
from typing import Dict

import pandas as pd

from classifier.util import timing
from .groupcounter import GroupCounter


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

        self.polarity_counter: Dict[str, GroupCounter] = {}

        logging.info(f'> Init Freq. Classifier, n-grams: {list(self.config["ngrams"].keys())}')

    #  -------- fit -----------
    #
    @timing
    def fit(self, data: pd.DataFrame, target_label: str, log_label: str = '***') -> None:
        logging.info(f'> Fit Freq. Classifier on {log_label}')

        for n, keep in self.config['ngrams'].items():
            self.polarity_counter[n] = GroupCounter(
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

        # calculate the scores
        for n, lookup in self.polarity_counter.items():
            lookup.predict(data, f'{n}-gram')

    #  -------- export -----------
    #
    def export(self, path: str):
        for n, lookup in self.polarity_counter.items():
            lookup.write(f'{path}{n}-gram-weights')

    #  -------- property -----------
    #
    @property
    def col_names(self):
        return [
            f'{n}-gram_{label}'
            for n, label in list(
                itertools.product(
                    self.polarity_counter.keys(),
                    self.target_values
                )
            )
        ]
