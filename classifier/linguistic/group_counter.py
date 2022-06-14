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

        self.analysis: dict = self.__create(
            self.data,
            self.key_label,
            self.group_label
        )

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
    def __create(self, data: pd.DataFrame, key_label: str, group_label: str) -> dict:

        analysis: dict = GroupCounter.calculate_absolute_frequencies(
            data,
            key_label=key_label,
            group_label=group_label,
            value_label=LABEL['abs_freq']
        )

        if self.config.get('pre_selection', self.default_config['pre_selection']) != -1:
            analysis = GroupCounter.get_most_common(
                analysis,
                LABEL['abs_freq'],
                self.config.get('pre_selection', self.default_config['pre_selection'])
            )

        if self.config.get('shared', self.default_config['shared']) == "subtract":
            GroupCounter.subtract_shared(analysis)

        elif self.config.get('shared', self.default_config['shared']) == "remove":
            analysis = GroupCounter.remove_shared(analysis)

        if self.config.get('post_selection', self.default_config['post_selection']) != -1:
            analysis = GroupCounter.get_most_common(
                analysis,
                LABEL['abs_freq'],
                self.config.get('post_selection', self.default_config['post_selection'])
            )

        GroupCounter.calculate_relative_frequencies(analysis, LABEL['abs_freq'], LABEL['rel_freq'])

        return analysis

    #  -------- subtract_shared -----------
    #
    @staticmethod
    def subtract_shared(analysis: dict):
        subtracted: dict = {}

        for sentiment, _ in analysis.items():
            for other_sentiment, other_count in analysis.items():
                if other_sentiment is not sentiment:
                    subtracted[sentiment] = analysis[sentiment].subtract(other_count, fill_value=0)

        return subtracted

    #  -------- predict -----------
    #
    def predict(self, data: pd.DataFrame, label: str) -> None:

        #  -------- predict -----------
        #
        def pred(row):
            return count.loc[count.index.isin(row)][LABEL['rel_freq']].sum()

        for group, count in self.analysis.items():
            data[f'{label}_{group}'] = data[label].apply(pred)

            # ToDo: enable parallel processing
            # data[f'{label}_{group}'] = data[label].parallel_apply(pred)

    #
    #
    #  -------- calculate_absolute_frequencies -----------
    #
    @staticmethod
    def calculate_absolute_frequencies(data: pd.DataFrame, key_label: str, group_label: str, value_label: str) -> dict:

        #  -------- calc_abs -----------
        #
        def calc_abs(df: pd.DataFrame) -> pd.DataFrame:
            #
            count = df[key_label].explode().value_counts().to_frame()

            count.rename({key_label: value_label}, axis='columns', inplace=True)

            return count

        return {label: calc_abs(group) for label, group in data.groupby(group_label)}

    #
    #
    #  -------- remove_shared -----------
    #
    @staticmethod
    def remove_shared(data: dict):
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
    #  -------- get_most_common -----------
    #
    @staticmethod
    def get_most_common(data: dict, column: str, num: int = 1024) -> dict:
        return {
            sentiment: count.sort_values(by=[column], ascending=False).head(num)
            for sentiment, count in data.items()
        }

    #
    #
    #  -------- calculate_relative_frequencies -----------
    #
    @staticmethod
    def calculate_relative_frequencies(data: dict, abs_freq_label: str, rel_freq_label: str) -> None:
        for sentiment, count in data.items():
            count[rel_freq_label] = count[abs_freq_label] / sum(count[abs_freq_label])

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
