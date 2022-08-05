import pandas as pd
import torch

from classifier.features.__main__ import Main as Runner
from classifier.util import timing
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

        # load and encode metacritic information
        self.metacritic = pd.read_csv(self.config['model']['hybrid']['metacritic_path'])
        self.encoder.df_encode(self.metacritic, col='summary', label=self.config['model']['hybrid']['metacritic_path'])

        # match metacritic into datasets
        for data_label, dataset in self.data.items():
            self.match(dataset.data)
            dataset.data[['sentiment', 'metacritic']].to_csv(f'{self.config["out_path"]}meta.{data_label}.csv')

    #  -------- __call__ -----------
    #
    def __call__(self, *args) -> None:
        super(
            type(self).__bases__[0], self
        ).__call__(
            int(self.encoder.dim) + len(self.pipeline.col_names) + 1,
            self.__collation_fn
        )

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        return (
            torch.concat([
                self.collate_encoder(batch),
                self.collate_features(batch),
                self.collate_hybrid(batch)
            ], dim=1),
            self.collate_target_label(batch)
        )

    #
    #
    #  -------- collate_hybrid -----------
    #
    @staticmethod
    def collate_hybrid(batch: list) -> torch.Tensor:
        return torch.stack([
            (
                torch.tensor(
                    sample[['metacritic']].values,
                    device=get_device()
                )
                .squeeze()
                .float()
            ) for sample in batch
        ])

    #
    #
    #  -------- match -----------
    #
    @timing
    def match(self, data: pd.DataFrame) -> None:
        pool = torch.stack(self.metacritic[self.encoder.col_name].tolist()).float().to(get_device())

        data["metacritic"] = data.apply(
            lambda row: self.metacritic.iloc[
                torch.norm(
                    pool - row[self.encoder.col_name].to(get_device()).unsqueeze(0),
                    dim=1
                )
                .argmin()
                .item()
            ]['metascore'],
            axis=1
        )


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
