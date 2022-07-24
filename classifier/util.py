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
    """Load JSON file."""
    with open(path) as data:
        return json.load(data)


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


#
#
#  -------- byte_to_mb -----------
#
def byte_to_mb(byte: int) -> str:
    return f'{byte / (1024.0 * 1024.0):2.4f} MB'
