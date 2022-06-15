import itertools
import logging
from typing import Dict

import pandas as pd

from classifier.lib.util import timing
from classifier.linguistic.group_counter import GroupCounter


class Model:

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config
        self.polarity_counter: Dict[str, GroupCounter] = {}

        logging.info(f'> Init Freq. Classifier, n-grams: {list(self.config["ngrams"].keys())}')

    #  -------- fit -----------
    #
    @timing
    def fit(self, data: pd.DataFrame, label: str = '***') -> None:
        logging.info(f'> Fit Freq. Classifier on {label}')

        for n in self.config['ngrams']:
            self.polarity_counter[n] = GroupCounter(
                data,
                key_label=f'{n}-gram',
                group_label=self.config["group_label"],
                config=self.config["ngrams"][str(n)]
            )

    #
    #
    #  -------- predict -----------
    #
    @timing
    def predict(self, data: pd.DataFrame, label: str = '***') -> None:
        logging.info(f'> Predict with Freq. Classifier on {label}')

        # calculate the scores
        for n, lookup in self.polarity_counter.items():
            lookup.predict(data, f'{n}-gram')

        # calculate sum for each label
        for label in self.config['polarities']:
            data[f"sum_{label}"] = data.filter(regex=f".*_{label}").sum(axis='columns')

        # get highest label by sum
        data['prediction'] = data.filter(regex=f"sum_.*").idxmax(axis="columns").str.replace('sum_', '')

    #  -------- save -----------
    #
    def save(self, path: str):
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
                    self.config['ngrams'],
                    self.config['polarities']
                )
            )
        ]
