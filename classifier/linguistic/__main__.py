import logging

import pandas as pd

from classifier import Data, Metric
from classifier.linguistic.model import Model
from classifier.util import load_config, load_logger


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
            'train': Data(self.config['data']['train_path'],
                          generate_token=True, generate_ngrams=self.config['model']['ngrams']),
            'eval': Data(self.config['data']['eval_path'],
                         generate_token=True, generate_ngrams=self.config['model']['ngrams']),
        }

        self.metric = Metric(self.logger)
        self.model = Model(self.config['model'])

    #
    #
    #
    #  -------- __call__ -----------
    def __call__(self):

        self.model.fit(self.data['train'].data)

        for n, lookup in self.model.polarities.items():
            lookup.write(f'{self.config["data"]["out_path"]}{n}-gram-weights')

        # predict train and eval set
        prediction: dict = {
            'train': self.model.predict(self.data['train'].data),
            'eval': self.model.predict(self.data['eval'].data),
        }

        labels: set = {'positive', 'negative'}

        # print results to console
        for data_label, data in prediction.items():
            self.metric.reset()

            for cat in labels:
                # create confusing matrix values for each category (omitting true negative)
                tps: int = sum(pd.Series((data['prediction'] == data['gold']) & (data['gold'] == cat)))
                fns: int = sum(pd.Series(data['gold'] == cat)) - tps
                fps: int = sum(pd.Series(data['prediction'] == cat)) - tps

                self.metric.add_tp(cat, tps)
                self.metric.add_fn(cat, fns)
                self.metric.add_fp(cat, fps)

                # add for every other class category matches to true negative
                for nc in (labels - {cat}):
                    self.metric.add_tn(nc, tps)

            self.metric.show()


if __name__ == "__main__":
    Main()()
