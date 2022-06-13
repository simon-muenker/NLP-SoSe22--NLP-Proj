import collections
import json
import logging

from functools import wraps

from time import time


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
#  -------- timing -----------
#
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        logging.info(f'> f({f.__name__}) took: {time() - start:2.4f} sec')

        return result

    return wrap


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
