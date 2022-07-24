import torch

from classifier.features.__main__ import Main as Runner


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

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
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
