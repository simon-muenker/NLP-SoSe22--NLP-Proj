import argparse
import collections
import json
import logging


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as data:
        return json.load(data)


#
#
#  -------- load_json -----------
#
def save_json(path: str, data: dict) -> None:
    """Save JSON configuration file."""
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


#
#
#  -------- dict_merge -----------
#
def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None

    # Copyright (C) 2016 Paul Durivage <pauldurivage+github@gmail.com>
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


#
#
#  -------- load_config -----------
#
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
