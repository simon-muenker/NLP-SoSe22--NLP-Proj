from classifier import Runner


class Main(Runner):

    #  -------- __init__ -----------
    #
    def __init__(self):
        super().__init__()

    #  -------- __call__ -----------
    #
    def __call__(self, *args) -> None:
        super().__call__(self.encoder.dim, self.__collation_fn)

    #
    #
    #  -------- __collation_fn -----------
    #
    def __collation_fn(self, batch: list) -> tuple:
        return (
            self.collate_encoder(batch),
            self.collate_target_label(batch)
        )


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
