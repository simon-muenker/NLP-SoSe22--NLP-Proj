from dataclasses import dataclass, field

from tqdm import tqdm

import nltk
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


@dataclass
class Data(Dataset):
    data_path: str
    generate_token: bool = False
    generate_ngrams: list = None
    data_frame: pd.DataFrame = field(init=False)

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        self.data = pd.read_csv(self.data_path)
        self.label_mapping: dict = {
            "positive": 1,
            "negative": 0
        }

        if self.generate_token:
            self.tokenize()

        if self.generate_ngrams:
            for n in self.generate_ngrams:
                self.ngrams(n)

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

    #  -------- tokenize -----------
    #
    def tokenize(self, label: str = 'token') -> None:
        tqdm.pandas(desc=f'Tokenize: {self.data_path}')

        # remove punctuation & html tags, convert to lowercase, tokenize
        self.data[label] = self.data['review'].progress_apply(
            lambda sent: nltk.tokenize.word_tokenize(sent, language='english')
        )

    def ngrams(self, n: int, label: str = 'token'):
        tqdm.pandas(desc=f'Generate {n}-gram: {self.data_path}')

        self.data[f'{n}-gram'] = self.data[label].progress_apply(
            lambda sent: nltk.ngrams(sent, n)
        )