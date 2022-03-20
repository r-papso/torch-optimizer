import random
from copy import deepcopy
from typing import Any, Iterable, Tuple

from torch import nn

from ..prune.pruner import Pruner


def prune_model(model: nn.Module, pruner: Pruner, mask: Any) -> nn.Module:
    model_cpy = deepcopy(model)
    model_cpy = pruner.prune(model_cpy, mask)
    return model_cpy


def mut_triangular(
    individual: Any, low: Iterable[int], up: Iterable[int], indpb: float
) -> Tuple[Any]:
    size = len(individual)

    for i, l, u in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = int(random.triangular(low=l, high=u, mode=individual[i]))

    return (individual,)
