from typing import Dict

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from classifier.linguistic.group_counter import GroupCounter


pandarallel.initialize(progress_bar=True, verbose=1)


class Model:

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config
        self.polarity_counter: Dict[str, GroupCounter] = {}

    #  -------- fit -----------
    #
    def fit(self, data: pd.DataFrame) -> None:

        for idx, n in enumerate(tqdm(self.config['ngrams'], desc="Fit LookUpDicts")):
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
    def predict(self, data: pd.DataFrame) -> None:

        # calculate the scores for ea
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
