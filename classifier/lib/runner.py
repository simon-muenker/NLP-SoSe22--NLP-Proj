import random
import argparse
import logging
from abc import abstractmethod

import torch

from classifier.lib import Data
from classifier.lib.util import dict_merge, load_json
from classifier.lib.neural.util import set_cuda_device


class Runner:

    #  -------- __init__ -----------
    #
    def __init__(self):
        # --- ---------------------------------
        # --- base setup
        self.config: dict = Runner.load_config()
        self.logger: logging = Runner.load_logger(self.config['out_path'] + "full.log")

        # --- ---------------------------------
        # --- load data
        self.logger.info(f"\n[--- LOAD/PREPARE DATA -> (train/eval/test) ---]")
        self.data: dict = self.load_data()

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

    #
    #
    #  -------- load_logger -----------
    #
    @staticmethod
    def load_logger(path: str, debug: bool = False):
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format="%(message)s",
            handlers=[
                logging.FileHandler(path, mode="w"),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    #  -------- __setup_pytorch -----------
    #
    def __setup_pytorch(self):
        # make pytorch computations deterministic
        # src: https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_cuda_device(self.config['cuda'])

        print(torch.cuda.current_device())
        exit()
