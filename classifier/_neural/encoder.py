import logging as logger
from typing import Tuple, List, Union

import torch
from transformers import AutoTokenizer, AutoModel, logging

from classifier.util import timing, dict_merge
from .util import get_device, unpad


class Encoder:

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "model": "bert-base-uncased",
            "layers": [10]
        }

    #  -------- __init__ -----------
    #
    @timing
    def __init__(self, user_config: dict = None):
        logging.set_verbosity_error()

        self.config: dict = Encoder.default_config()
        dict_merge(self.config, user_config)

        logger.info(f'> Init Encoder: \'{self.config["model"]}\'')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = AutoModel.from_pretrained(self.config["model"], output_hidden_states=True).to(get_device())

    #
    #
    #  -------- batch_encode -----------
    #
    @torch.no_grad()
    def batch_encode(
            self,
            batch: List[str],
            return_unpad: bool = True
    ) -> Union[Tuple[List, List[torch.Tensor], list], Tuple[List, torch.Tensor, torch.Tensor]]:
        encoding = self.tokenizer(batch, padding=True, truncation=True)

        tokens: List[List[str]] = [self.ids_to_tokens(ids) for ids in encoding['input_ids']]
        ids: torch.Tensor = torch.tensor(encoding['input_ids'], dtype=torch.long, device=get_device())
        masks: torch.Tensor = torch.tensor(encoding['attention_mask'], dtype=torch.short, device=get_device())
        ctes: torch.Tensor = self.contextualize(ids, masks)

        if return_unpad:
            masks = masks.sum(1)

            return (
                unpad(tokens, masks),
                unpad(ctes, masks),
                unpad(ids, masks)
            )

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
                self.config["layers"],
                dtype=torch.int32,
                device=get_device()
            )
        ).sum(1).squeeze()

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
