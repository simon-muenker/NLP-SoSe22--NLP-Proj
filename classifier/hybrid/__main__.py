import torch

from classifier.features.__main__ import Main as Runner
from .model import Model
from .._neural.util import get_device


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

    #  -------- __call__ -----------
    #
    def __call__(self, model=None, collation_fn=None):

        model = Model(
            in_size=tuple([
                int(self.encoder.dim),
                len(self.pipeline.col_names)
            ]),
            out_size=len(self.data['train'].get_label_keys()),
            config=self.config['model']
        )

        super(Runner, self).__call__(model, self.__collation_fn)

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []
        pipeline: list = []

        # collate data
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

        # embed text
        _, sent_embeds, _ = self.encoder(text, return_unpad=False)

        return (
            (sent_embeds[:, 1], torch.stack(pipeline)),
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
