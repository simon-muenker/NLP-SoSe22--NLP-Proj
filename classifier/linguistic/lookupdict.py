from collections import Counter

import pandas as pd


class LookUpDict:

    #
    #
    #  -------- __init__ -----------
    def __init__(
            self,
            config: dict
    ) -> None:
        self.config = config

        self.data: dict = self.fit()

    #
    #
    #
    # -------- fit -----------
    def fit(self):

        abs_freq: dict = LookUpDict.calculate_absolute_frequencies(
            self.config['data'], group_label=self.config['group_label'], token_label=self.config['token_label'])

        pre_most_common: dict = LookUpDict.get_most_common(abs_freq, 'n', self.config['pre_selection'])

        wo_shared: dict = LookUpDict.remove_shared(pre_most_common, token_label='token')

        fin_most_common: dict = LookUpDict.get_most_common(wo_shared, 'n', self.config['final_selection'])

        LookUpDict.calculate_relative_frequencies(fin_most_common)

        return fin_most_common

    #
    #
    #
    # -------- calculate_absolute_frequencies -----------
    @staticmethod
    def calculate_absolute_frequencies(data: pd.DataFrame, group_label: str, token_label: str) -> dict:
        return {
            # count most common words (absolute frequencies)
            label: LookUpDict._calc_abs(group[token_label])
            # iterate over each sentiment
            for label, group in data.groupby(group_label)
        }

    #
    #
    #
    #
    @staticmethod
    def remove_shared(data: dict, token_label: str):
        #
        token_intersection: set = set.intersection(
            *map(set, [list(count[token_label]) for _, count in data.items()]))
        #
        return {
            sentiment: count[~count[token_label].isin(token_intersection)]
            for sentiment, count in data.items()
        }

    #
    #
    #
    # -------- _calc_abs -----------
    @staticmethod
    def _calc_abs(data: pd.Series, key_label: str = 'token', value_label: str = 'n') -> pd.DataFrame:
        return pd.DataFrame.from_records(
            list(dict(Counter(list(data.explode()))).items()),
            columns=[key_label, value_label]
        )

    #
    #
    #
    # -------- get_most_common -----------
    @staticmethod
    def get_most_common(data: dict, column: str, num: int = 1024) -> dict:
        return {
            sentiment: count.sort_values(by=[column], ascending=False).head(num)
            for sentiment, count in data.items()
        }

    #
    #
    #
    # -------- calculate_relative_frequencies -----------
    @staticmethod
    def calculate_relative_frequencies(data: dict) -> None:
        for sentiment, count in data.items():
            count['p'] = count['n'] / sum(count["n"])

    #
    #
    #
    # -------- __repr__ -----------
    def __repr__(self) -> str:
        return "".join(
            f'\nname({sentiment}) || len({len(count)}) || sum({sum(count["n"])}) \n {count.head(4)}'
            for sentiment, count in self.data.items()
        )

    #
    #
    #
    # -------- write -----------
    def write(self, path: str) -> None:
        writer = pd.ExcelWriter(str(path) + ".xlsx")

        for i, (label, df) in enumerate(self.data.items()):
            df.to_excel(writer, label)
        writer.save()
