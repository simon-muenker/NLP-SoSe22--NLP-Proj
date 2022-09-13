import collections
import json
import logging
from functools import wraps
from time import time
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
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
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recourses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into ``dct``.

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
    return f'{byte / (1024.0 ** 2):2.4f} MB'


#
#
#  -------- get_device -----------
#
def get_device() -> str:
    return f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'


#
#
#  -------- unpad -----------
#
def unpad(padded: Union[list, torch.Tensor], length: Union[list, torch.Tensor]) -> Union[list, torch.Tensor]:
    """Convert the given packaged sequence into a list of vectors."""
    output = []
    for v, n in zip(padded, length):
        output.append(v[:n])
    return output


#
#
#  -------- load_iterator -----------
#
def load_iterator(
        data: Dataset,
        collate_fn: callable = lambda x: x,
        batch_size: int = 8,
        shuffle: bool = False,
        num_workers: int = 0,
        desc: str = "",
        disable: bool = False):
    return enumerate(tqdm(DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ),
        leave=False,
        desc=desc,
        disable=disable,
    ))


#
#
#  -------- load_iterator -----------
#
def memory_usage(model: torch.nn.Module) -> int:
    return sum([
        sum([param.nelement() * param.element_size() for param in model.parameters()]),
        sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    ])
