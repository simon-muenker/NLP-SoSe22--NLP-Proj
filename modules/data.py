import logging
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from .util import timing, byte_to_mb


@dataclass
class Data(Dataset):
    file_path: str
    data_label: str
    target_label: str
    target_groups: dict

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        self.data: pd.DataFrame = Data.__load(self.file_path)
        logging.info((
            f'> Load/Init from {self.file_path}\n'
            f'  Number of Samples: {len(self)} \n'
            f'  Memory Usage: {byte_to_mb(self.data.memory_usage(deep=True).sum())}'
        ))

    #  -------- load -----------
    #
    @staticmethod
    @timing
    def __load(file_path: str):
        return pd.read_csv(file_path)

    #  -------- save -----------
    #
    @timing
    def save(self, file_path: str):
        logging.info(f'> Save to {file_path}.csv')
        self.data.to_csv(f'{file_path}.csv')

    #  -------- encode_target_label -----------
    #
    def encode_target_label(self, label: str) -> int:
        return self.target_groups.get(label)

    #  -------- decode_target_label -----------
    #
    def decode_target_label(self, label: int) -> str:
        return {v: k for k, v in self.target_groups.items()}.get(label)

    #  -------- get_target_label_keys -----------
    #
    def get_target_label_keys(self) -> set:
        return set(k for k in self.target_groups.keys())

    #  -------- get_target_label_values -----------
    #
    def get_target_label_values(self) -> set:
        return set(k for k in self.target_groups.values())

    #  -------- __getitem__ -----------
    #
    def __getitem__(self, idx) -> T_co:
        return self.data.iloc[[idx]]

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)
