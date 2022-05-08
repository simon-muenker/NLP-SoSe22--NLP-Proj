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

        # tokenize data and generate ngrams
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
        self.model.save(self.config['out_path'])

        # predict train and eval set
        prediction: dict = {
            'train': self.model.predict(self.data['train'].data),
            'eval': self.model.predict(self.data['eval'].data),
        }

        # print results to console
        for data_label, data in prediction.items():
            self.logger.info(f"\n[--- EVAL -> {data.data_path} ---]")
            self.metric.reset()
            self.metric.confusion_matrix(
                self.data['train'].get_label_keys(),
                'prediction', 'gold', data)
            self.metric.show()


if __name__ == "__main__":
    Main()()
