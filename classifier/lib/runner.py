import argparse
import logging
from abc import abstractmethod

from classifier.lib import Data
from classifier.lib.util import dict_merge, load_json


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
            'train': Data(self.config['data']['train_path'], polarities=self.config['data']['polarities'],
                          config=self.config['data']['options']),
            'eval': Data(self.config['data']['eval_path'], polarities=self.config['data']['polarities'],
                         config=self.config['data']['options']),
            'test': Data(self.config['data']['test_path'], polarities=self.config['data']['polarities'],
                         config=self.config['data']['options'])
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
