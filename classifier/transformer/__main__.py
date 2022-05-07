import logging
import random

import torch

from classifier import Data
from classifier.util import load_config, load_logger

from classifier.transformer.encoding import Encoding
from classifier.transformer.model import Model
from classifier.transformer.trainer import Trainer
from classifier.transformer.util import get_device


class Main:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):

        # --- ---------------------------------
        # --- base setup
        self.config: dict = load_config()
        self.logger: logging = load_logger(self.config['data']['out_path'] + "full.log")
        self.setup_pytorch()

        # --- ---------------------------------
        # --- load components

        # load data
        self.data: dict = {
            'train': Data(self.config['data']['train_path']),
            'eval': Data(self.config['data']['eval_path']),
            'test': Data(self.config['data']['test_path'])
        }

        # load encoding, model
        self.encoding = Encoding(self.config['encoding'])
        self.model = Model(
            in_size=self.encoding.dim,
            out_size=2,
            config=self.config['model']
        ).to(get_device())

        # load trainer
        self.trainer = Trainer(
            self.model,
            self.data,
            self.collation_fn,
            logger=self.logger,
            out_dir=self.config['data']['out_path'],
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
    #  -------- setup_pytorch -----------
    #
    def setup_pytorch(self):
        # make pytorch computations deterministic
        # src: https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
