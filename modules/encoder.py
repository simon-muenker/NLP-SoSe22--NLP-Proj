import logging as logger
import os
from typing import Tuple, List, Union, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging

from .util import timing, dict_merge, byte_to_mb, get_device, unpad, memory_usage

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Encoder:

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            'model': 'bert-base-uncased',
            'layers': [10]
        }

    #  -------- __init__ -----------
    #
    @timing
    def __init__(self, user_config: dict = None):
        logging.set_verbosity_error()

        self.config: dict = Encoder.default_config()
        dict_merge(self.config, user_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'])
        self.model = AutoModel.from_pretrained(self.config['model'], output_hidden_states=True).to(get_device())

        logger.info((
            f'> Init Encoder: \'{self.config["model"]}\'\n'
            f'  Memory Usage: {byte_to_mb(memory_usage(self.model))}'
        ))

    #
    #
    #  -------- batch_encode -----------
    #
    @torch.no_grad()
    def batch_encode(
            self,
            batch: List[str],
            return_unpad: bool = True
    ) -> Tuple[Union[list, Tensor], ...]:
        encoding = self.tokenizer(batch, padding=True, truncation=True)

        tokens: List[List[str]] = [self.ids_to_tokens(ids) for ids in encoding['input_ids']]
        ids: torch.Tensor = torch.tensor(encoding['input_ids'], dtype=torch.long, device=get_device())
        masks: torch.Tensor = torch.tensor(encoding['attention_mask'], dtype=torch.short, device=get_device())
        ctes: torch.Tensor = self.contextualize(ids, masks)

        if return_unpad:
            return tuple(unpad(el, masks.sum(1)) for el in [tokens, ctes, ids])

        return tokens, ctes, ids

    #  -------- contextualize -----------
    #
    @torch.no_grad()
    def contextualize(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.index_select(
            torch.stack(
                self.model.forward(ids, masks).hidden_states,
                dim=1
            ),
            dim=1,
            index=torch.tensor(
                self.config['layers'],
                dtype=torch.int32,
                device=get_device()
            )
        ).sum(1).squeeze()

    #  -------- df_encode -----------
    #
    @timing
    def df_encode(
            self,
            data: pd.DataFrame,
            col: str,
            form: Literal['cls', 'mean'] = 'mean',
            batch_size: int = 32,
            label: str = '***'
    ):
        embeds: list = []

        extraction: dict = {
            'cls': lambda x: x[:, 1].cpu(),
            'mean': lambda x: torch.mean(x, dim=1)
        }

        for _, group in tqdm(
                data.groupby(np.arange(len(data)) // batch_size),
                leave=False, desc=f'Encode {label}'
        ):
            # extract text content from dataframe group
            content: list = list(group[col].values)

            # add dimension if last element is one dimensional
            if len(content) == 1:
                content.append('')

            # batch forward content
            _, batch_embeds, _ = self(content, return_unpad=False)

            # extract sentence vector
            embeds.extend(torch.unbind(extraction.get(form, 'mean')(batch_embeds).cpu()))

        # pop if batch was padded
        if len(data) % batch_size == 1:
            embeds.pop()

        data[self.col_name] = embeds

    #  -------- ids_to_tokens -----------
    #
    def ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    #  -------- ids_to_sent -----------
    #
    def ids_to_sent(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    #  -------- dim -----------
    #
    @property
    def dim(self) -> int:
        return self.model.config.to_dict()['hidden_size']

    #  -------- __call__ -----------
    #
    def __call__(self, batch: List[str], return_unpad: bool = True):
        return self.batch_encode(batch, return_unpad=return_unpad)

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return self.model.config.to_dict()['vocab_size']

    #  -------- col_name -----------
    #
    @property
    def col_name(self):
        return 'prediction'
