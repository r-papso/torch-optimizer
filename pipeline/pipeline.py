import torch.nn as nn
from context import Context

from ..pruning.pruner import Pruner
from ..training.trainer import Trainer


class Pipeline:
    def __init__(self, model: nn.Module, pruner: Pruner, trainer: Trainer) -> None:
        self.__pruner = pruner
        self.__trainer = trainer
        self.__context = Context()
        self.__context.model = model

    def run(self):
        self.__trainer.before_pruning_train(self.__context)

        for _ in range(self.__pruner.n_steps()):
            self.__pruner.prune(self.__context)
            self.__trainer.during_pruning_train(self.__context)

        self.__trainer.after_pruning_train(self.__context)

