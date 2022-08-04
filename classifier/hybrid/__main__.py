import pandas as pd
import torch

from classifier.features.__main__ import Main as Runner


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
            self.match(dataset)
            print(dataset)
            exit()

    #  -------- __call__ -----------
    #
    def __call__(self, *args) -> None:
        super(
            type(self).__bases__[0], self
        ).__call__(
            int(self.encoder.dim) + len(self.pipeline.col_names),
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
                self.collate_features(batch)
            ], dim=1),
            self.collate_target_label(batch)
        )

    #
    #
    #  -------- match -----------
    #
    def match(self, data: pd.DataFrame, col: str) -> None:
        pool = torch.stack(self.metacritic['prediction'].tolist()).float()

        data["metacritic"] = data['prediction'].parallel_apply(
            lambda vector: self.metacritic.iloc[
                torch.norm(pool - vector.unsqueeze(0), dim=1).argmin().item()
            ]['metascore'])


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
