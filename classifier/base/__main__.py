import torch

from classifier import Runner
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
            in_size=self.encoder.dim,
            out_size=len(self.data['train'].get_label_keys()),
            config=self.config['model']['neural']
        )

        super().__call__(model, self.__collation_fn)

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []

        # collate data
        for _, review, sentiment in batch:
            print(review)
            text.append(review)
            label.append(sentiment)

        # embed text
        _, sent_embeds, _ = self.encoder(text, return_unpad=False)

        # extract only first embeddings (CLS); transform labels
        return (
            sent_embeds[:, 1],
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
