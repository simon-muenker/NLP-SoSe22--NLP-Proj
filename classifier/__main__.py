import argparse
import logging
import random

import torch

from classifier import Data
from classifier.util import dict_merge, load_json, byte_to_mb
from ._neural import Trainer, Encoder, Model
from ._neural.util import get_device


class Main:

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            'seed': 1,
            'cuda': 0,
            'debug': False,
            'out_path': './__',
            'data': {},
            'model': {
                'neural': Model.default_config(),
            },
            'trainer': Trainer.default_config()
        }

    #  -------- __init__ -----------
    #
    def __init__(self):

        # --- ---------------------------------
        # --- base setup
        self.config: dict = Main.default_config()
        dict_merge(self.config, Main.__load_config())

        Main.__load_logger(f'{self.config["out_path"]}full.log')
        Main.__setup_pytorch(self.config["seed"], self.config["cuda"])

        # --- ---------------------------------
        # --- load data
        logging.info(f'\n[--- PREPARE DATA -> ({list(k for k in self.config["data"]["paths"].keys())}) ---]')
        self.data: dict = self.load_data()

        # --- ---------------------------------
        # --- load components
        logging.info('\n[--- LOAD COMPONENTS ---]')

        # encoder
        self.encoder = None
        if self.config['model'].get('encoding', False):
            self.encoder = Encoder(self.config['model']['encoding'])

            for _, dataset in self.data.items():
                self.encoder.df_encode(dataset.data, col=dataset.data_label, label=dataset.data_path)
                logging.info(f'> Memory Usage (w/ Embeds): {byte_to_mb(dataset.data.memory_usage(deep=True).sum())}')

    #  -------- __call__ -----------
    #
    def __call__(self) -> None:
        self.config['model']['neural']['in_size'] = self.encoder.dim
        self.config['model']['neural']['out_size'] = len(self.data['train'].get_label_keys())

        model = Model(self.config['model']['neural'])

        Trainer(
            model, self.data, self.collation_fn,
            out_dir=self.config['out_path'],
            user_config=self.config['trainer'],
        )()

    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        return (
            self.collate_encoder(batch),
            self.collate_target_label(batch)
        )

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self) -> dict:
        return {
            name: Data(
                data_path=path,
                polarities=self.config['data']['polarities'],
                data_label=self.config['data']['data_label'],
                target_label=self.config['data']['target_label']
            ) for name, path in self.config['data']['paths'].items()
        }

    #
    #
    #  -------- collate_target_label -----------
    #
    def collate_target_label(self, batch: list) -> torch.Tensor:
        return torch.tensor(
            [self.data['train'].encode_label(lb) for lb in
             [sample[self.data['train'].target_label].values[0] for sample in batch]],
            dtype=torch.long, device=get_device()
        )

    #
    #
    #  -------- collate_encoder -----------
    #
    def collate_encoder(self, batch: list) -> torch.Tensor:
        return torch.stack([
            sample[self.encoder.col_name].values[0]
            for sample in batch
        ]).to(get_device())

    #
    #
    #  -------- __load_config -----------
    #
    @staticmethod
    def __load_config() -> dict:
        # get console arguments, config file
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-C',
            dest='config',
            nargs='+',
            required=True,
            help='define multiple config.json files witch are combined during runtime (overwriting is possible)'
        )
        args = parser.parse_args()

        config_collection: dict = {}

        for config in args.config:
            dict_merge(config_collection, load_json(config))

        return config_collection

    #
    #
    #  -------- __load_logger -----------
    #
    @staticmethod
    def __load_logger(path: str, debug: bool = False) -> None:
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format='%(message)s',
            handlers=[
                logging.FileHandler(path, mode='w'),
                logging.StreamHandler()
            ]
        )
        logging.info(f'> Loaded logger: {path}')

    #  -------- __setup_pytorch -----------
    #
    @staticmethod
    def __setup_pytorch(seed: int, cuda: int) -> None:

        # check if cuda is available
        if not torch.cuda.is_available():
            cuda = None

        # set cuda to last device
        elif cuda == -1 or cuda > torch.cuda.device_count():
            cuda = torch.cuda.device_count() - 1

        # make pytorch computations deterministic
        logging.info(f'> Setup PyTorch: seed({seed}), cuda({cuda})')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(cuda) if cuda else None


#
#
#  -------- __main__ -----------
#
if __name__ == '__main__':
    Main()()
