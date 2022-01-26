from pyacc.pruning.model.abstract import ModelPruner
from pyacc.pruning.scoring.abstract import Scoring
from torch import nn


class GlobalPruner(ModelPruner):
    def __init__(self) -> None:
        super().__init__()

    def prune(self, model: nn.Module, scoring: Scoring, strategy: str, shrink_model: bool) -> None:
        # TODO implement
        pass
