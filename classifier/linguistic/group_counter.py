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

    config: dict = None
    analysis: dict = None

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        # load default config file if is None
        if self.config is None:
            self.config = self.default_config

        self.analysis: dict = self.__create()

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            "pre_selection": 1024,
            "shared": "remove",
            "post_selection": 512
        }

    #
    #
    #  -------- __create -----------
    #
    def __create(self) -> dict:

        analysis: dict = GroupCounter.__calc_abs_freq(
            self.data, key_label=self.key_label, group_label=self.group_label
        )

        analysis = GroupCounter.__most_common(analysis, self.config['pre_selection'])

        if self.config['shared'] == "subtract":
            GroupCounter.__subtract_shared(analysis)

        elif self.config['shared'] == "remove":
            analysis = GroupCounter.__remove_shared(analysis)

        analysis = GroupCounter.__most_common(analysis, self.config['post_selection'])

        GroupCounter.__calc_rel_freq(analysis)

        return analysis

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

    #  -------- __subtract_shared -----------
    #
    @staticmethod
    def __subtract_shared(analysis: dict):
        subtracted: dict = {}

        for sentiment, _ in analysis.items():
            for other_sentiment, other_count in analysis.items():
                if other_sentiment is not sentiment:
                    subtracted[sentiment] = analysis[sentiment].subtract(other_count, fill_value=0)

        return subtracted

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
    def __calc_rel_freq(data: dict) -> None:
        for sentiment, count in data.items():
            count[LABEL['rel_freq']] = count[LABEL['abs_freq']] / sum(count[LABEL['abs_freq']])

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
