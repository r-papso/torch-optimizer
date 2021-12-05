from torch import nn
from torch.utils.data import Dataset

from ..pruning.abstract import Pruner
from ..training.abstract import Trainer


class Pipeline:
    def __init__(self, pruner: Pruner, trainer: Trainer) -> None:
        self.__pruner = pruner
        self.__trainer = trainer

    def run(self):
        self.__trainer.before_pruning_train()

        for _ in range(self.__pruner.n_steps()):
            self.__pruner.prune()
            self.__trainer.during_pruning_train()

        self.__trainer.after_pruning_train()

