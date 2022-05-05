from dataclasses import dataclass, field

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


@dataclass
class Data(Dataset):
    data_path: str
    data_frame: pd.DataFrame = field(init=False)

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        self.data = pd.read_csv(self.data_path)
        self.label_mapping: dict = {
            "positive": 1,
            "negative": 0
        }

    #  -------- __getitem__ -----------
    #
    def __getitem__(self, idx) -> T_co:
        return self.data['review'][idx], self.data['sentiment'][idx]

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)

    #  -------- encode_label -----------
    #
    def encode_label(self, label: str) -> int:
        return self.label_mapping.get(label)

    #  -------- decode_label -----------
    #
    def decode_label(self, label: int) -> str:
        return {v: k for k, v in self.label_mapping.items()}.get(label)
