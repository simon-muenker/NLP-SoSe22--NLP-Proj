import logging
from dataclasses import dataclass

import pandas as pd
from nela_features.nela_features import NELAFeatureExtractor


@dataclass
class NELAPipe:
    config: dict = None

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            'style': True,
            'complexity': True,
            'bias': True,
            'affect': True,
            'moral': True,
            'event': True

        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        if self.config is None:
            self.config = self.default_config

        self.pipeline = NELAFeatureExtractor()

        self.mapping: dict = {
            'style': self.pipeline.extract_style,
            'complexity': self.pipeline.extract_complexity,
            'bias': self.pipeline.extract_bias,
            'affect': self.pipeline.extract_affect,
            'moral': self.pipeline.extract_moral,
            'event': self.pipeline.extract_event
        }

        self.col_names = self.__get_cols()
        logging.info(f'> Init NELA Pipeline')

    #
    #
    #  -------- apply -----------
    #
    def apply(self, data: pd.DataFrame, col: str) -> None:
        #  -------- __apply -----------
        #
        def __apply(row: str):
            feature_vector: list = []

            for feat, fn in self.mapping.items():
                if self.config[feat]:
                    vector, _ = fn(row)
                    feature_vector.extend(vector)

            return pd.Series(feature_vector, index=self.col_names)

        data[self.col_names] = data[col].parallel_apply(__apply)

    #  -------- __get_cols -----------
    #
    def __get_cols(self, from_sample: str = 'Colorless green ideas sleep furiously') -> list:
        col_labels: list = []

        for feat, fn in self.mapping.items():
            if self.config[feat]:
                _, col = fn(from_sample)
                col_labels.extend(col)

        return col_labels
