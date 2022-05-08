import math
from typing import Dict

from pandarallel import pandarallel
from tqdm import tqdm

import pandas as pd

from classifier.linguistic.lookupdict import LookUpDict

pandarallel.initialize(progress_bar=True, verbose=1)


class Model:

    #
    #
    #  -------- __init__ -----------
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config

        self.polarities: Dict[str, LookUpDict] = {}

    #
    #
    #
    # -------- fit -----------
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
    #
    # -------- predict -----------
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:

        # create empty predictions dataframe
        predictions: pd.DataFrame = pd.DataFrame()

        # calculate a score for each polarity
        for n, lookup in self.polarities.items():
            target_label: str = 'token' if n == 1 else f'{n}-gram'

            for label, count in lookup.data.items():

                predictions[f'{n}-gram_{label}'] = data[target_label].parallel_apply(
                    lambda x: Model.calc_score(x, count) * math.log(int(n)))

        predictions["sum_positive"] = predictions.filter(regex=".*_positive").sum(axis='columns')
        predictions["sum_negative"] = predictions.filter(regex=".*_negative").sum(axis='columns')

        predictions['prediction'] = predictions.apply(
            lambda row: 'positive' if row["sum_positive"] > row["sum_negative"] else 'negative', axis=1
        )

        # add gold labels
        predictions['gold'] = data['sentiment']

        return predictions

    @staticmethod
    def calc_score(token: list, count: pd.DataFrame) -> float:
        score: float = 0.0

        for t in token:
            val = count.loc[count['token'] == t, 'p']

            if not val.empty:
                score += val.iloc[0]

        return score
