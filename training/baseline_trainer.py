from training.abstract import Trainer


class BaselineTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()

    def before_pruning_train(self) -> None:
        pass

    def during_pruning_train(self) -> None:
        pass

    def after_pruning_train(self) -> None:
        pass
