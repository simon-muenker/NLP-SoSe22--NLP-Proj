import logging
import re
from dataclasses import dataclass, field

import nltk
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from classifier.util import timing, dict_merge

LABEL: dict = {
    'token': '1-gram',
    'grams': '-gram'
}


@dataclass
class Data(Dataset):
    data_path: str
    polarities: dict

    data_label: str
    target_label: str

    data_language: str = 'english'
    user_config: dict = field(default_factory=dict)

    stop_words: list = field(default_factory=list)

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            'postprocess': True,
            'ngrams': [1, 2, 3],
            'remove_stopwords': True
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        self.config = Data.default_config()
        dict_merge(self.config, self.user_config)

        self.data: pd.DataFrame = Data.__load(self.data_path)
        logging.info((
            f'> Load/Init from {self.data_path}\n'
            f'  Number of Samples: {len(self)} \n'
            f'  Memory Usage: {self.data.memory_usage(deep=True).sum() / (1024.0 * 1024.0):2.4f} MB'
        ))

        if self.config['remove_stopwords']:
            self.stop_words = list(nltk.corpus.stopwords.words(self.data_language))

        if self.config['postprocess']:
            self.postprocess()

    #  -------- postprocess -----------
    #
    def postprocess(self) -> None:

        # tokenize
        self.__tokenize()

        # generate ngrams
        if self.config['ngrams']:
            for n in self.config['ngrams']:
                if n > 1:
                    self.__ngram(n)

    #  -------- __tokenize -----------
    #
    @timing
    def __tokenize(self) -> None:

        #  -------- tok -----------
        #
        def tok(sent: str):
            # convert to lowercase, trim
            sent: str = sent.lower().strip()

            # remove non-alphabetical characters
            sent: str = re.sub('[^a-zA-Z]', ' ', sent)

            # tokenize with TreebankWordTokenizer
            token: list = nltk.tokenize.word_tokenize(sent, language=self.data_language)

            if self.config['remove_stopwords']:
                token: list = [t for t in token if t not in self.stop_words]

            return token

        self.data[LABEL['token']] = self.data[self.data_label].parallel_apply(tok)

    #  -------- __ngram -----------
    #
    @timing
    def __ngram(self, n: int) -> None:
        self.data[f'{n}{LABEL["grams"]}'] = self.data[LABEL['token']].parallel_apply(
            lambda sent: list(nltk.ngrams(sent, n))
        )

    #  -------- load -----------
    #
    @staticmethod
    @timing
    def __load(path):
        return pd.read_csv(path)

    #  -------- save -----------
    #
    @timing
    def save(self, path: str):
        logging.info(f'> Save to {path}.csv')
        self.data.to_csv(f'{path}.csv')

    #  -------- encode_label -----------
    #
    def encode_label(self, label: str) -> int:
        return self.polarities.get(label)

    #  -------- decode_label -----------
    #
    def decode_label(self, label: int) -> str:
        return {v: k for k, v in self.polarities.items()}.get(label)

    #  -------- get_label_keys -----------
    #
    def get_label_keys(self) -> set:
        return set(k for k in self.polarities.keys())

    #  -------- get_label_values -----------
    #
    def get_label_values(self) -> set:
        return set(k for k in self.polarities.values())

    #  -------- __getitem__ -----------
    #
    def __getitem__(self, idx) -> T_co:
        return (
            self.data.iloc[[idx]],
            self.data[self.data_label][idx],
            self.data[self.target_label][idx]
        )

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)
