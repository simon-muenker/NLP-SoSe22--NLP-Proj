import logging

from classifier import Runner, Metric
from .model import Model


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        super().__init__()
        self.metric = Metric()
        self.model = Model(self.config['model'])

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):
        logging.info("\n[--- RUN ---]")

        self.model.fit(self.data['train'].data, label=self.data['train'].data_path)
        self.model.save(self.config['out_path'])

        # predict train, eval
        logging.info(f'\n[--- EVAL -> {self.config["data"]["eval_on"]} ---]')
        for data_label, dataset in self.data.items():

            if data_label not in self.config["data"]["eval_on"]:
                continue

            # predict dataset
            self.model.predict(dataset.data, label=dataset.data_path)
            self.metric.reset()
            self.metric.confusion_matrix(
                dataset.get_label_keys(),
                dataset.data['sentiment'],
                dataset.data['prediction']
            )
            self.metric.show()
            self.metric.export(f'{self.config["out_path"]}metric.{data_label}')


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
