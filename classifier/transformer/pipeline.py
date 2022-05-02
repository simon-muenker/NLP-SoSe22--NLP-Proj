import argparse
import logging
import random

import torch

from classifier import Data
from classifier.util import dict_merge, load_json

from .encoding import Encoding
from .model import Model
from .trainer import Trainer
from .util import get_device


class TransformerPipeline:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):

        # --- ---------------------------------
        # --- base setup
        self.config: dict = self.load_config()
        self.logger: logging = self.setup_logging()
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
        )

        # load trainer
        self.trainer = Trainer(
            self.model,
            self.data['train'],
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
        # --- run trainer
        self.logger.info("\n[--- TRAINING ---]")
        self.logger.info(f"Train on: {self.data['train'].data_path}: ")
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
        cls_embeds: list = [tco[0].to(get_device()) for tco in sent_embeds]

        # label mapping (refactor to config, or auto generate)
        label_mapping: dict = {
            'positive': 0,
            'negative': 1,
        }

        # transform labels
        label_ids: list = [
            torch.tensor([label_mapping.get(lb)], dtype=torch.long, device=get_device())
            for lb in label
        ]

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
    #  -------- setup_logging -----------
    #
    def setup_logging(self):
        filename: str = self.config['data']['out_path'] + "full.log"

        logging.basicConfig(
            level=logging.INFO if not self.config["debug"] else logging.DEBUG,
            format="%(message)s",
            handlers=[
                logging.FileHandler(filename, mode="w"),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    #
    #
    #  -------- load_config -----------
    #
    @staticmethod
    def load_config() -> dict:
        # get console arguments, config file
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-C",
            dest="config",
            nargs='+',
            required=True,
            help="define multiple config.json files witch are combined during runtime (overwriting is possible)"
        )
        args = parser.parse_args()

        config_collection: dict = {}

        for config in args.config:
            dict_merge(config_collection, load_json(config))

        return config_collection
