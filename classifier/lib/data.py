import logging
import re
from dataclasses import dataclass, field

import nltk
import pandas as pd
from pandarallel import pandarallel
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from classifier.lib.util import timing

pandarallel.initialize(progress_bar=True, verbose=1)

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
    config: dict = None

    stop_words: list = field(default_factory=list)
    lemmatizer: nltk.stem.WordNetLemmatizer = None

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            "postprocess": True,
            "ngrams": [1, 2],
            "remove_stopwords": True
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        logging.info(f'> Load/Init from {self.data_path}')

        if self.config is None:
            self.config = self.default_config

        self.data: pd.DataFrame = Data.__load(self.data_path)

        if self.config['remove_stopwords']:
            self.stop_words = list(nltk.corpus.stopwords.words(self.data_language))

        if self.config['postprocess']:
            self.postprocess()

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

            # remove html tags
            sent: str = re.sub("(<[^>]+>)", '', sent)

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
