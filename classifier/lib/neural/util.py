
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


#
#
#  -------- get_device -----------
#
def get_device(i: int = 0) -> str:
    return f'cuda:{i}' if torch.cuda.is_available() else 'cpu'


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
