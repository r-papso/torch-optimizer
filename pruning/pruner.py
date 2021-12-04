from abc import ABC, abstractmethod

from ..pipeline.context import Context


class Pruner(ABC):
    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.__n_steps = n_steps

    def n_steps(self) -> int:
        return self.__n_steps

    @abstractmethod
    def prune(self, context: Context) -> None:
        pass