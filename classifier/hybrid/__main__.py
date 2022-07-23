import torch

from classifier.features.__main__ import Main as Runner
from .._neural.util import get_device


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
        text: list = []
        label: list = []
        pipeline: list = []

        # collate features
        for sample, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)
            pipeline.append(
                torch.tensor(
                    sample[self.pipeline.col_names].values,
                    device=get_device()
                )
                .squeeze()
                .float()
            )

        # compute embeddings
        _, sent_embeds, _ = self.encoder(text, return_unpad=False)

        return (
            torch.concat([
                # sent_embeds[:, 1], extract CLS
                torch.mean(sent_embeds, dim=1),
                torch.stack(pipeline)
            ], dim=1),
            torch.tensor(
                [self.data['train'].encode_label(lb) for lb in label],
                dtype=torch.long, device=get_device()
            )
        )


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
