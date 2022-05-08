import torch

from classifier.lib import Runner

from classifier.hybrid import Model as Network
from classifier.linguistic import Model as Classifier
from classifier.transformer import Encoding

from classifier.lib.neural import Trainer
from classifier.lib.neural.util import get_device


class Main(Runner):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # --- ---------------------------------
        # --- load components

        # tokenize data and generate ngrams
        for _, dataset in self.data.items():
            dataset.tokenize()
            for n in self.config['model']['ngrams']:
                dataset.ngrams(n)

        # load encoding, neural, classifier
        self.encoding = Encoding(self.config['model']['encoding'])
        self.net = Network(
            in_size=tuple([756, 2]),
            out_size=2,
            config=self.config['model']['neural']
        )
        self.clss = Classifier(self.config['model']['linguistic'])

        # load trainer
        self.trainer = Trainer(
            self.net,
            self.data,
            self.collation_fn,
            logger=self.logger,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

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
        # --- fit, save, predict classifier
        self.clss.fit(self.data['train'].data)
        self.clss.save(self.config['out_path'])

        # predict train and eval set
        prediction: dict = {
            'train': self.clss.predict(self.data['train'].data),
            'eval': self.clss.predict(self.data['eval'].data),
        }

        # concat predictions and datasets

        # --- ---------------------------------
        # --- train
        self.trainer()

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []

        # collate data
        for sample in batch:
            text.append(sample[0])
            label.append(sample[1])

        # embed text
        _, sent_embeds, _ = self.encoding(text)

        # extract only first embeddings (CLS)
        cls_embeds: list = [tco[0] for tco in sent_embeds]

        # transform labels
        label_ids: torch.Tensor = torch.tensor(
            [self.data['train'].encode_label(lb) for lb in label],
            dtype=torch.long, device=get_device()
        )

        return cls_embeds, label_ids


if __name__ == "__main__":
    Main()()
