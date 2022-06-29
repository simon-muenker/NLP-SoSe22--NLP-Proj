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
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self, ):
        if self.config is None:
            self.config = self.default_config

        self.pipeline = NELAFeatureExtractor()
        logging.info(f'> Init NELA Pipeline')

    #
    #
    #  -------- apply -----------
    #
    def apply(self, data: pd.DataFrame, col: str) -> None:

        #  -------- __apply -----------
        #
        def __apply(row: str):
            vector, labels = self.pipeline.extract_all(row)
            return pd.Series(vector, index=labels)

        data[self.col_names] = data[col].apply(__apply)

    #  -------- col_names -----------
    #
    @property
    def col_names(self):
        _, labels = self.pipeline.extract_all('Colorless green ideas sleep furiously')
        return labels
