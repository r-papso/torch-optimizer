from __future__ import annotations

from typing import Any

from torch import nn

from ..prune.pruner import Pruner
from . import utils


class Cache:
    """Class provides caching of last pruned model.
    
    Caching last pruned model is useful in cases when multiple constraints or objective
    functions are present in the optimization problem modelling neural network pruning.
    """

    __model = None
    __solution = None

    @classmethod
    def get_pruned_model(cls, model: nn.Module, pruner: Pruner, solution: Any) -> nn.Module:
        """Gets pruned model according to original model, pruner and solution produced by
        optimization. If the same solution is provided repeatedly, chached model is returned.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            solution (Any): Solution representing mask, which defines pruning structure.

        Returns:
            nn.Module: Pruned model.
        """
        if solution is not None and solution is cls.__solution:
            return cls.__model

        cls.__solution = solution
        cls.__model = utils.prune_model(model, pruner, solution)

        return cls.__model
