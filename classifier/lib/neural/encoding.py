import logging as logger
from typing import Tuple, List

import torch
from transformers import AutoTokenizer, AutoModel, logging

from classifier.lib.neural.util import get_device
from classifier.lib.util import timing
from classifier.transformer.util import unpad


class Encoding:

    #  -------- __init__ -----------
    #
    @timing
    def __init__(self, config: dict = None):
        logging.set_verbosity_error()

        if config is None:
            config = self.default_config()

        self.config = config

        logger.info(f'> Init Encoder: \'{self.config["model"]}\'')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.model = AutoModel.from_pretrained(self.config["model"], output_hidden_states=True).to(get_device())

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "model": "bert-base-uncased",
            "layers": [-1]
        }

    #
    #
    #  -------- batch_encode -----------
    #
    @torch.no_grad()
    def batch_encode(
            self,
            batch: List[str],
            return_unpad: bool = True
    ) -> Tuple[List, List[torch.Tensor], list]:
        encoding = self.tokenizer(batch, padding=True, truncation=True)

        tokens: List[List[str]] = [self.ids_to_tokens(ids) for ids in encoding['input_ids']]
        ids: torch.Tensor = torch.tensor(encoding['input_ids'], device=get_device()).long()
        masks: torch.Tensor = torch.tensor(encoding['attention_mask'], device=get_device()).short()
        ctes: torch.Tensor = self.contextualize(ids, masks)

        if return_unpad:
            masks = masks.sum(1)

            return (
                unpad(tokens, masks),
                unpad(ctes, masks),
                unpad(ids, masks)
            )

        return tokens, list(ctes), list(ids)

    #  -------- contextualize -----------
    #
    @torch.no_grad()
    def contextualize(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.model.forward(ids, masks).hidden_states[i]
             for i in self.config["layers"]]
        ).sum(0).squeeze()

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
    def __call__(self, batch: List[str]):
        return self.batch_encode(batch)

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return self.model.config.to_dict()['vocab_size']
