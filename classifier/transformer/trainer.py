import csv
import logging
from datetime import datetime
from typing import Callable

import torch
from torch import optim
from torch.utils.data import Dataset

from .util import load_iterator


class Trainer:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            model: torch.nn.Module,
            data: Dataset,
            collation_fn: Callable,
            logger: logging,
            out_dir: str,
            config: dict = None):

        self.state: dict = {
            'epoch': [],
            'train_loss': [],
            'duration': [],
        }

        self.model = model
        self.data = data
        self.collation_fn = collation_fn

        self.logger = logger
        self.out_dir = out_dir

        # load config file else use default
        if config is None:
            config = self.default_config()

        self.config = config

        # setup loss_fn, optimizer, scheduler and early stopping
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), **self.config["optimizer"])

    #
    #
    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            "epochs": 25,
            "shuffle": True,
            "batch_size": 256,
            "num_workers": 0,
            "report_rate": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "betas": [
                    0.9,
                    0.98
                ],
                "eps": 1e-9
            }
        }

    #
    #
    #  -------- __call__ (train) -----------
    #
    def __call__(self) -> dict:
        saved_model_epoch: int = 0

        # --- epoch loop
        try:
            for epoch in range(1, self.config["epochs"] + 1):
                time_begin: datetime = datetime.now()

                # --- ---------------------------------
                # --- begin train
                train_loss: float = 0.0
                for idx, batch in load_iterator(
                        self.data,
                        collate_fn=self.collation_fn,
                        batch_size=self.config["batch_size"],
                        shuffle=self.config["shuffle"],
                        num_workers=self.config["num_workers"],
                        desc=f"Train, epoch: {epoch:03}",
                        disable=epoch % self.config["report_rate"] != 0
                ):
                    train_loss = self._train(batch, idx, train_loss)

                # --- ---------------------------------
                # --- update state
                self.state["epoch"].append(epoch)
                self.state["train_loss"].append(train_loss)
                self.state["duration"].append(datetime.now() - time_begin)

                # --- ---------------------------------
                # --- save if is best model
                if self.state["train_loss"][-1] < min(n for n in self.state["train_loss"] if n > 0):
                    saved_model_epoch = self.state["epoch"][-1]
                    self.model.save(self.out_dir + "model.bin")

                # --- ---------------------------------
                # --- log to user
                if epoch % self.config["report_rate"] == 0:
                    self._log(epoch)

        except KeyboardInterrupt:
            self.logger.warning("Warning: Training interrupted by user!")

        # load last save model
        self.logger.info("Load best model based on evaluation loss.")
        self.model = self.model.load(self.out_dir + "model.bin")
        self._log(saved_model_epoch)

        # return and write train state to main
        self._write_state()
        return self.state

    #
    #
    #  -------- _train -----------
    #
    def _train(self, batch: dict, batch_id: int, train_loss: float) -> float:
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        loss = self.model.train_step(self.loss_fn, batch)
        loss.backward()

        # scaling the gradients down, places a limit on the size of the parameter updates
        # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])

        # optimizer step
        self.optimizer.step()

        # save loss, acc for statistics
        train_loss += (loss.item() - train_loss) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        return train_loss

    #
    #
    #  -------- _log -----------
    #
    def _log(self, epoch: int) -> None:
        self.logger.info((
            f"@{epoch:03}: \t"
            f"loss(train)={self.state['train_loss'][epoch - 1]:2.5f} \t"
            f"duration(epoch)={self.state['duration'][epoch - 1]}"
        ))

    #
    #
    #  -------- _write_state -----------
    #
    def _write_state(self) -> None:
        cols: list = list(self.state.keys())

        with open(self.out_dir + 'train.csv', 'w') as output_file:
            writer = csv.writer(output_file, delimiter=",")
            writer.writerow(cols)
            writer.writerows(zip(*[self.state[c] for c in cols]))
