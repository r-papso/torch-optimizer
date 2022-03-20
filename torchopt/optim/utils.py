from copy import deepcopy
from typing import Any
from torch import nn

from ..prune.pruner import Pruner


def prune_model(model: nn.Module, pruner: Pruner, mask: Any) -> nn.Module:
    model_cpy = deepcopy(model)
    model_cpy = pruner.prune(model_cpy, mask)
    return model_cpy
