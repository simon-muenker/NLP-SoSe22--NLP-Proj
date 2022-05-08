from classifier.lib import Runner

from classifier.hybrid import Model as Network
from classifier.linguistic import Model as Classifier
from classifier.transformer import Encoding


class Main(Runner):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # --- ---------------------------------
        # --- load components

        # load encoding, model
        # self.encoding = Encoding(self.config['model']['encoding'])
        self.net = Network(
            in_size=tuple([756, 2]),
            out_size=2,
            config=self.config['model']['neural']
        )

        for _, dataset in self.data.items():
            dataset.tokenize()
            for n in self.config['model']['ngrams']:
                dataset.ngrams(n)

        self.clss = Classifier(self.config['model']['linguistic'])

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):
        # --- ---------------------------------
        # --- init
        self.logger.info("\n[--- INIT ---]")
        self.logger.info(f"- Model has {len(self.net)} trainable parameters.")

        # --- ---------------------------------
        # --- fit, predict classifier
        self.clss.fit(self.data['train'].data)

        for n, lookup in self.clss.polarities.items():
            lookup.write(f'{self.config["data"]["out_path"]}{n}-gram-weights')

        # predict train and eval set
        prediction: dict = {
            'train': self.clss.predict(self.data['train'].data),
            'eval': self.clss.predict(self.data['eval'].data),
        }

        # --- ---------------------------------
        # --- train, eval neural model


if __name__ == "__main__":
    Main()()
