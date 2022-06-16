import argparse
import logging
import random
from abc import abstractmethod


import torch
from pandarallel import pandarallel

from classifier.lib import Data
from classifier.lib.util import dict_merge, load_json

pandarallel.initialize(progress_bar=True)


class Runner:

    #  -------- __init__ -----------
    #
    def __init__(self):
        # --- ---------------------------------
        # --- base setup
        self.config: dict = Runner.__load_config()
        Runner.__load_logger(f'{self.config["out_path"]}full.log')
        Runner.__setup_pytorch(self.config["seed"], self.config["cuda"])

        # --- ---------------------------------
        # --- load data
        logging.info(f'\n[--- PREPARE DATA -> ({list(k for k in self.config["data"]["paths"].keys())}) ---]')
        self.data: dict = self.load_data()

        logging.info('\n[--- LOAD COMPONENTS ---]')

    #  -------- __call__ -----------
    #
    @abstractmethod
    def __call__(self):
        pass

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
                target_label=self.config['data']['target_label'],
                config=self.config['data']['config']
            ) for name, path in self.config['data']['paths'].items()
        }

    #
    #
    #  -------- __load_config -----------
    #
    @staticmethod
    def __load_config() -> dict:
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

    #
    #
    #  -------- __load_logger -----------
    #
    @staticmethod
    def __load_logger(path: str, debug: bool = False) -> None:
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format="%(message)s",
            handlers=[
                logging.FileHandler(path, mode="w"),
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

        else:
            # set cuda to last device
            if cuda == -1 or cuda > torch.cuda.device_count():
                cuda = torch.cuda.device_count() - 1

        # make pytorch computations deterministic
        logging.info(f'> Setup PyTorch: seed({seed}), cuda({cuda})')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(cuda) if cuda else None
