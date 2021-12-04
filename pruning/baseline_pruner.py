from pipeline.context import Context
from pruning.pruner import Pruner


class BaselinePruner(Pruner):
    def __init__(self, n_steps: int) -> None:
        super().__init__(n_steps)

    def prune(self, context: Context) -> None:
        pass
