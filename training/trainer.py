from abc import ABC, abstractmethod

from ..pipeline.context import Context


class Trainer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def before_pruning_train(self, context: Context) -> None:
        pass

    @abstractmethod
    def during_pruning_train(self, context: Context) -> None:
        pass

    @abstractmethod
    def after_pruning_train(self, context: Context) -> None:
        pass
