import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from torch import optim

from . import Metric, Data
from .util import dict_merge, load_iterator


@dataclass
class Trainer:
    model: torch.nn.Module
    data: Dict[str, Data]
    collation_fn: Callable
    out_dir: str
    user_config: dict = field(default_factory=dict)

    #  -------- default_config -----------
    #
    @staticmethod
    def default_config() -> dict:
        return {
            'epochs': 100,
            'shuffle': True,
            'batch_size': 32,
            'num_workers': 0,
            'report_rate': 5,
            'max_grad_norm': 2.0,
            'optimizer': {
                'lr': 1e-3,
                'weight_decay': 1e-2,
                'betas': [
                    0.9,
                    0.999
                ],
                'eps': 1e-8
            }
        }

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        self.config = self.default_config()
        dict_merge(self.config, self.user_config)

        self.state: dict = {
            'epoch': [],
            'loss_train': [],
            'loss_test': [],
            'f1_train': [],
            'f1_test': [],
            'duration': [],
        }

        # setup loss_fn, optimizer, scheduler and early stopping
        self.metric = Metric()
        self.optimizer = optim.AdamW(self.model.parameters(), **self.config['optimizer'])

    #
    #
    #  -------- __call__ (train) -----------
    #
    def __call__(self) -> dict:
        logging.info(f'\n[--- TRAIN -> {self.data["train"].file_path} ---]')

        saved_model_epoch: int = 0
        saved_test_metric: tuple = ()

        # --- epoch loop
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                time_begin: datetime = datetime.now()

                # --- ---------------------------------
                # --- begin train
                self.metric.reset()
                loss_train: float = 0.0
                for idx, batch in load_iterator(
                        self.data['train'],
                        collate_fn=self.collation_fn,
                        batch_size=self.config['batch_size'],
                        shuffle=self.config['shuffle'],
                        num_workers=self.config['num_workers'],
                        desc=f'Train, epoch: {epoch:03}',
                        disable=epoch % self.config['report_rate'] != 0
                ):
                    loss_train = self._train(batch, idx, loss_train)
                f1_train: float = self.metric.f_score()

                # --- ---------------------------------
                # --- begin eval
                self.metric.reset()
                loss_test: float = 0.0
                for idx, batch in load_iterator(
                        self.data['test'],
                        collate_fn=self.collation_fn,
                        batch_size=self.config['batch_size'],
                        shuffle=self.config['shuffle'],
                        num_workers=self.config['num_workers'],
                        desc=f'Eval, epoch: {epoch:03}',
                        disable=epoch % self.config['report_rate'] != 0
                ):
                    loss_test = self._eval(batch, idx, loss_test)
                f1_test: float = self.metric.f_score()

                # --- ---------------------------------
                # --- update state
                self.state['epoch'].append(epoch)
                self.state['loss_train'].append(loss_train)
                self.state['loss_test'].append(loss_test)
                self.state['f1_train'].append(f1_train)
                self.state['f1_test'].append(f1_test)
                self.state['duration'].append(datetime.now() - time_begin)

                # --- ---------------------------------
                # --- save if is best model
                if self.state['f1_test'][-1] >= max(n for n in self.state['f1_test'] if n > 0):
                    saved_model_epoch = self.state['epoch'][-1]
                    self.model.save(self.out_dir + 'model.bin')
                    saved_test_metric = self.metric.save()

                # --- ---------------------------------
                # --- log to user
                if epoch % self.config['report_rate'] == 0:
                    self._log(epoch)

        except KeyboardInterrupt:
            logging.warning('> Warning: Training interrupted by user!')

        # load last save model
        logging.info('> Load best model based on evaluation loss.')
        self.model = self.model.load(self.out_dir + 'model.bin')
        self._log(saved_model_epoch)

        # return and write train state to main
        self._write_state()

        # --- ---------------------------------
        # --- eval
        logging.info(f'\n[--- TEST -> {self.data["test"].file_path} ---]')
        self.metric.load(saved_test_metric)
        self.metric.show(decoding=self.data['train'].decode_target_label)
        self.metric.export(f'{self.out_dir}metric.test', decoding=self.data['train'].decode_target_label)

        return self.state

    #
    #
    #  -------- _train -----------
    #
    def _train(self, batch: tuple, batch_id: int, loss_train: float) -> float:
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        loss, pred_labels = self.model.train_step(batch)
        loss.backward()

        # scaling the gradients down, places a limit on the size of the parameter updates
        # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])

        # optimizer step
        self.optimizer.step()

        # save loss, acc for statistics
        loss_train += (loss.item() - loss_train) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        # calculate metric
        self._metric(batch, pred_labels)

        return loss_train

    #
    #
    #  -------- _eval -----------
    #
    def _eval(self, batch: tuple, batch_id: int, loss_eval: float) -> float:
        self.model.eval()

        loss, pred_labels = self.model.train_step(batch)

        # save loss, acc for statistics
        loss_eval += (loss.item() - loss_eval) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        # calculate metric
        self._metric(batch, pred_labels)

        return loss_eval

    #
    #
    #  -------- _metric -----------
    #
    def _metric(self, batch: Tuple[list, torch.Tensor], pred_labels: torch.Tensor) -> None:
        _, gold_labels = batch

        self.metric.confusion_matrix(
            self.data['train'].get_target_label_values(),
            pd.Series(gold_labels.cpu().numpy()), pd.Series(pred_labels.cpu().numpy())
        )

    #  -------- _log -----------
    #
    def _log(self, epoch: int) -> None:
        logging.info((
            f'@{epoch:03}: \t'
            f"loss(train)={self.state['loss_train'][epoch - 1]:2.4f} \t"
            f"loss(test)={self.state['loss_test'][epoch - 1]:2.4f} \t"
            f"f1(train)={self.state['f1_train'][epoch - 1]:2.4f} \t"
            f"f1(test)={self.state['f1_test'][epoch - 1]:2.4f} \t"
            f"duration(epoch)={self.state['duration'][epoch - 1]}"
        ))

    #  -------- _write_state -----------
    #
    def _write_state(self) -> None:
        cols: list = list(self.state.keys())

        with open(self.out_dir + 'metric.training.csv', 'w') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            writer.writerow(cols)
            writer.writerows(zip(*[self.state[c] for c in cols]))
