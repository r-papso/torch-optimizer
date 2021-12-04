from pipeline.context import Context
from training.trainer import Trainer


class BaselineTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()

    def before_pruning_train(self, context: Context) -> None:
        pass

    def during_pruning_train(self, context: Context) -> None:
        pass

    def after_pruning_train(self, context: Context) -> None:
        pass
