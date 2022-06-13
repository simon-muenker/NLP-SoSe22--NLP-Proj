import torch

from classifier.lib import Runner
from classifier.lib.neural import Encoding, Trainer
from classifier.lib.neural.util import get_device
from classifier.transformer import Model


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # --- ---------------------------------
        # --- load components

        # encoding, model
        self.encoding = Encoding(self.config['model']['encoding'])
        self.model = Model(
            in_size=self.encoding.dim,
            out_size=2,
            config=self.config['model']['neural']
        ).to(get_device())

        # trainer
        self.trainer = Trainer(
            self.model,
            self.data,
            self.collation_fn,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

    #  -------- __call__ -----------
    #
    def __call__(self):
        self.trainer()

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []

        # collate data
        for _, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)

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
