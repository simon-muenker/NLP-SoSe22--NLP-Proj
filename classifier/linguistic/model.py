from typing import Dict

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from classifier.linguistic.lookupdict import LookUpDict

pandarallel.initialize(progress_bar=True, verbose=1)


class Model:

    #  -------- __init__ -----------
    #
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config
        self.polarities: Dict[str, LookUpDict] = {}

    #  -------- fit -----------
    #
    def fit(self, data: pd.DataFrame) -> None:

        for idx, n in enumerate(tqdm(self.config['ngrams'], desc="Fit LookUpDicts")):
            self.polarities[n] = LookUpDict({
                'data': data,
                'token_label': 'token' if n == 1 else f'{n}-gram',
                'group_label': 'sentiment',
                'pre_selection': self.config['pre_selection'][idx],
                'final_selection': self.config['final_selection'][idx]
            })

    #
    #
    #  -------- predict -----------
    #
    def predict(self, data: pd.DataFrame) -> None:

        # calculate a score for each polarity
        for n, lookup in self.polarities.items():
            target_label: str = 'token' if n == 1 else f'{n}-gram'

            for label, count in lookup.data.items():
                data[f'{n}-gram_{label}'] = data[target_label].parallel_apply(
                    lambda row: count.loc[count['token'].isin(row)]['p'].sum())

        # calculate sum for each label
        for _, lookup in self.polarities.items():
            for label, _ in lookup.data.items():
                data[f"sum_{label}"] = data.filter(regex=f".*_{label}").sum(axis='columns')

        # FIXME generic prediction function
        data['prediction'] = data.apply(
            lambda row: 'positive' if row["sum_positive"] > row["sum_negative"] else 'negative', axis=1
        )

    #  -------- save -----------
    #
    def save(self, path: str):
        for n, lookup in self.polarities.items():
            lookup.write(f'{path}{n}-gram-weights')
