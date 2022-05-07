from typing import Dict

import pandas as pd

from classifier.linguistic.lookupdict import LookUpDict


class Model:

    #
    #
    #  -------- __init__ -----------
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config

        self.polarity_words: Dict[str, dict] = {}

        # pd.read_excel(
        #             self.config['weights']['uni_gram'],
        #             sheet_name=self.config['polarities']
        #         )

    #
    #
    #
    # -------- fit -----------
    def fit(self, data: pd.DataFrame) -> None:

        for idx, n in enumerate(self.config['ngrams']):
            self.polarity_words[n] = LookUpDict({
                'data': data,
                'token_label': 'token' if n == 1 else f'{n}-gram',
                'group_label': 'sentiment',
                'pre_selection_num': 1024,
                'final_selection_num': 64
            }).data

    #
    #
    #
    # -------- predict -----------
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:

        # create empty predictions dataframe
        predictions: pd.DataFrame = pd.DataFrame()

        # calculate a score for each polarity
        for n, lookup in self.polarity_words.items():
            target_label: str = 'token' if n == 1 else f'{n}-gram'

            for label, count in lookup.items():
                predictions[f'{n}-gram_{label}'] = data[target_label].apply(lambda x: Model.calc_score(x, count))

        predictions["sum_positive"] = predictions.filter(regex=".*_positive").sum(axis='columns')
        predictions["sum_negative"] = predictions.filter(regex=".*_negative").sum(axis='columns')

        predictions['prediction'] = predictions.apply(
            lambda row: 'positive' if row["sum_positive"]> row["sum_negative"] else 'negative', axis=1
        )

        # add gold labels
        predictions['gold'] = data['sentiment']

        return predictions

    @staticmethod
    def calc_score(token: list, count: pd.DataFrame):
        score: float = 0.0

        for t in token:
            val = count.loc[count['token'] == t, 'p']

            if not val.empty:
                score += val.iloc[0]

        return score
