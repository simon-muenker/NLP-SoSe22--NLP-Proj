from dataclasses import dataclass

import pandas as pd

LABEL: dict = {
    'abs_freq': 'n',
    'rel_freq': 'p'
}


@dataclass
class GroupCounter:
    data: pd.DataFrame
    key_label: str
    group_label: str

    keep: int = 1024

    analysis: dict = None

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        gc = GroupCounter

        self.analysis = gc.__calc_abs_freq(self.data, self.key_label, self.group_label)
        self.analysis = gc.__most_common(self.analysis, self.keep)
        self.analysis = gc.__remove_shared(self.analysis)
        self.analysis = gc.__calc_rel_freq(self.analysis)

    #  -------- predict -----------
    #
    def predict(self, data: pd.DataFrame, label: str) -> None:
        for group, count in self.analysis.items():
            data[f'{label}_{group}'] = data[label].parallel_apply(
                lambda row: count.loc[count.index.isin(row)][LABEL['rel_freq']].sum()
            )

    #
    #
    #  -------- __calc_abs_freq -----------
    #
    @staticmethod
    def __calc_abs_freq(data: pd.DataFrame, key_label: str, group_label: str) -> dict:
        return {
            label: (
                group[key_label]
                .explode()
                .value_counts()
                .to_frame()
                .rename(
                    {key_label: LABEL['abs_freq']},
                    axis='columns'
                )
            ) for label, group in data.groupby(group_label)}

    #
    #
    #  -------- __remove_shared -----------
    #
    @staticmethod
    def __remove_shared(data: dict):
        #
        token_intersection: set = set.intersection(
            *map(set, [list(count.index) for _, count in data.items()]))
        #
        return {
            sentiment: count[~count.index.isin(token_intersection)]
            for sentiment, count in data.items()
        }

    #
    #
    #  -------- __most_common -----------
    #
    @staticmethod
    def __most_common(data: dict, num: int = 1024) -> dict:
        return {
            sentiment: (
                count
                .sort_values(
                    by=LABEL['abs_freq'],
                    ascending=False
                )
                .head(num)
            )
            for sentiment, count in data.items()
        }

    #
    #
    #  -------- __calc_rel_freq -----------
    #
    @staticmethod
    def __calc_rel_freq(data: dict) -> dict:
        return {
            sentiment: (
                count
                .assign(
                    p=count[LABEL['abs_freq']] / sum(count[LABEL['abs_freq']])
                )
            ) for sentiment, count in data.items()
        }

    #  -------- __repr__ -----------
    #
    def __repr__(self) -> str:
        return "".join(
            f'\nname({sentiment}) || len({len(count)}) || sum({sum(count["n"])}) \n {count.head(16)}'
            for sentiment, count in self.data.items()
        )

    #  -------- write -----------
    #
    def write(self, path: str) -> None:
        writer = pd.ExcelWriter(str(path) + ".xlsx")

        for i, (label, df) in enumerate(self.analysis.items()):
            df.to_excel(writer, label)
        writer.save()
