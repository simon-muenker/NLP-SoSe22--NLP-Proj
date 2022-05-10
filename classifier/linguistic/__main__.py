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

    #
    #
    #
    #  -------- __call__ -----------
    def __call__(self):

        self.model.fit(self.data['train'].data)
        self.model.save(self.config['out_path'])

        # predict train, eval
        for data_label, dataset in self.data.items():

            # skip test for now
            if data_label == 'test':
                continue

            # predict dataset
            self.logger.info(f"\n[--- PREDICT -> {dataset.data_path} ---]")
            self.model.predict(dataset.data)

            self.logger.info(f"[--- EVAL -> {dataset.data_path} ---]")
            self.metric.reset()
            self.metric.confusion_matrix(
                dataset.data, dataset.get_label_keys(),
                'prediction', 'sentiment')
            self.metric.show()


if __name__ == "__main__":
    Main()()
