import logging

import pandas as pd

from classifier import Data
from classifier.util import load_config, load_logger

from classifier.linguistic.model import Model


class Main:

    #
    #
    #
    #  -------- __init__ -----------
    def __init__(self) -> None:

        # --- ---------------------------------
        # --- base setup
        self.config: dict = load_config()
        self.logger: logging = load_logger(self.config['data']['out_path'] + "full.log")

        # --- ---------------------------------
        # --- load components
        self.data: dict = {
            'train': Data(self.config['data']['train_path'], generate_token=True, generate_ngrams=[2, 3]),
            'eval': Data(self.config['data']['eval_path'], generate_token=True, generate_ngrams=[2, 3]),
        }

        self.model = Model(self.config['model'])

    #
    #
    #
    #  -------- __call__ -----------
    def __call__(self):

        self.model.fit(self.data['train'].data)

        for name, lookup in self.model.polarities.items():
            lookup.write(self.config['data']['out_path'] + str(name))

        # predict train and eval set
        prediction: dict = {
            'train': self.model.predict(self.data['train'].data),
            'eval': self.model.predict(self.data['eval'].data),
        }

        # print results to console
        for data_label, data in prediction.items():
            # count valid predictions, print to console
            valid: int = sum(pd.Series(data['prediction'] == data['gold']))
            self.logger.info(f'ACC({data_label})\t= {valid / len(data):.4f}')


if __name__ == "__main__":
    Main()()

