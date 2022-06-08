import re
from dataclasses import dataclass, field

import nltk
import pandas as pd
from pandarallel import pandarallel
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


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
            "ngrams": [1, 2, 3],
            "remove_stopwords": True
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        # read data from csv file
        self.data: pd.DataFrame = pd.read_csv(self.data_path)

        # load default config file if is None
        if self.config is None:
            self.config = self.default_config

        if self.config['remove_stopwords']:
            self.stop_words = list(nltk.corpus.stopwords.words(self.data_language))

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
        self.tokenize()

        # generate ngrams
        if self.config['ngrams']:
            for n in self.config['ngrams']:
                self.ngrams(n)

    #  -------- tokenize -----------
    #
    def tokenize(self) -> None:

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

    #  -------- ngrams -----------
    #
    def ngrams(self, n: int) -> None:
        if n > 1:
            self.data[f'{n}{LABEL["grams"]}'] = self.data[LABEL['token']].parallel_apply(
                lambda sent: list(nltk.ngrams(sent, n))
            )

    #  -------- save -----------
    #
    def save(self, path: str):
        self.data.to_csv(path + ".csv")
