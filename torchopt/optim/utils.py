import random
from copy import deepcopy
from typing import Any, Iterable, Tuple

from torch import nn

from ..prune.pruner import Pruner


def prune_model(model: nn.Module, pruner: Pruner, mask: Any) -> nn.Module:
    """Helper function to prune model by pruner and mask provided as parameters.

    Args:
        model (nn.Module): Model to be pruned.
        pruner (Pruner): Pruner used for pruning the model.
        mask (Any): Mask defining pruning structure.

    Returns:
        nn.Module: Pruned model.
    """
    model_cpy = deepcopy(model)
    model_cpy = pruner.prune(model_cpy, mask)
    return model_cpy


def mut_triangular(
    individual: Any, low: Iterable[int], up: Iterable[int], indpb: float
) -> Tuple[Any]:
    """Implementation of mutation based on triangular probability distribution.

    Args:
        individual (Any): Individual to be mutated.
        low (Iterable[int]): List of lower bounds, len(low) == len(individual).
        up (Iterable[int]): List of upper bounds, len(up) == len(individual).
        indpb (float): Probability of mutating individual elements of the individual.

    Returns:
        Tuple[Any]: Mutated individual.
    """
    size = len(individual)

    for i, l, u in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = int(random.triangular(low=l, high=u + 1, mode=individual[i]))

    return (individual,)
