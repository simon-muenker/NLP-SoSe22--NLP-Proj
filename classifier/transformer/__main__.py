import torch

from classifier.lib import Runner
from classifier.lib.neural import Trainer

from classifier.transformer import Model
from classifier.transformer import Encoding
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

        # load encoding, model
        self.encoding = Encoding(self.config['encoding'])
        self.model = Model(
            in_size=self.encoding.dim,
            out_size=2,
            config=self.config['model']
        ).get_device()

        # load trainer
        self.trainer = Trainer(
            self.model,
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
        self.logger.info(f"- Model has {len(self.model)} trainable parameters.")

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


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
