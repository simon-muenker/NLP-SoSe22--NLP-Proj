import re
from dataclasses import dataclass, field

import nltk
import pandas as pd
from pandarallel import pandarallel
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

pandarallel.initialize(progress_bar=True, verbose=1)


@dataclass
class Data(Dataset):
    data_path: str
    polarities: dict
    generate_token: bool = False
    generate_ngrams: list = None
    remove_stopwords: bool = True
    data_language: str = 'english'
    stop_words: list = field(default_factory=list)

    def __post_init__(self):
        # read data from csv file
        self.data: pd.DataFrame = pd.read_csv(self.data_path)

        # generate token column
        if self.generate_token:
            self.tokenize()

        # generate ngrams
        if self.generate_ngrams:
            for n in self.generate_ngrams:
                self.ngrams(n)

    #  -------- __getitem__ -----------
    #
    def __getitem__(self, idx) -> T_co:
        return self.data.iloc[[idx]], self.data['review'][idx], self.data['sentiment'][idx]

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

    #  -------- tokenize -----------
    #
    def tokenize(self, label: str = 'token') -> None:

        # load stopwords from nltk
        if self.remove_stopwords:
            self.stop_words = list(nltk.corpus.stopwords.words(self.data_language))

        #  -------- __tokenize -----------
        #
        def __tokenize(sent: str):
            # convert to lowercase, trim
            sent: str = sent.lower().strip()

            # remove html tags
            sent: str = re.sub("(<[^>]+>)", '', sent)

            # remove non-alphabetical characters
            sent: str = re.sub('[^a-zA-Z]', ' ', sent)

            # tokenize with TreebankWordTokenizer
            token: list = nltk.tokenize.word_tokenize(sent, language=self.data_language)

            # remove stop words
            token: list = [t for t in token if t not in self.stop_words]

            return token

        self.data[label] = self.data['review'].parallel_apply(lambda sent: __tokenize(sent))

    #  -------- ngrams -----------
    #
    def ngrams(self, n: int, label: str = 'token'):
        if n != 1:
            self.data[f'{n}-gram'] = self.data[label].parallel_apply(
                lambda sent: list(nltk.ngrams(sent, n))
            )

    #  -------- save -----------
    #
    def save(self, path: str):
        self.data.to_csv(path + ".csv")
