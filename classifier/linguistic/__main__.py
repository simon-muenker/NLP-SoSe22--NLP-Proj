from classifier.lib import Runner, Metric
from classifier.linguistic import Model


class Main(Runner):

    #
    #
    #
    #  -------- __init__ -----------
    def __init__(self) -> None:
        super().__init__()

        self.metric = Metric(self.logger)
        self.model = Model(self.config['model'])

        for _, dataset in self.data.items():
            dataset.tokenize()

            for n in self.config['model']['ngrams']:
                dataset.ngrams(n)

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

        labels: set = self.data['train'].get_label_keys()

        # print results to console
        for data_label, data in prediction.items():
            self.metric.reset()
            self.metric.confusion_matrix(labels, 'prediction', 'gold', data)
            self.metric.show()


if __name__ == "__main__":
    Main()()
