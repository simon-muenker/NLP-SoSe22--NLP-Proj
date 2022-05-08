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

        self.model.predict(self.data['train'].data)
        self.model.predict(self.data['eval'].data)

        # print results to console
        for data_label, data in self.data.items():
            self.logger.info(f"\n[--- EVAL -> {self.data[data_label].data_path} ---]")
            self.metric.reset()
            self.metric.confusion_matrix(
                self.data['train'].get_label_keys(),
                'prediction', 'gold', data)
            self.metric.show()


if __name__ == "__main__":
    Main()()
