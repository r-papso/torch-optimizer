from __future__ import annotations

import torch.nn as nn


class Constraint:
    def __init__(self, previous: Constraint) -> None:
        self._prev = previous

    def feasible(self, model: nn.Module) -> bool:
        return self._prev.feasible(model) if self._prev is not None else True


class ChannelConstraint(Constraint):
    def __init__(self, previous: Constraint) -> None:
        super().__init__(previous)

    def feasible(self, model: nn.Module) -> bool:
        if not super().feasible(model):
            return False

        for module in model.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and any(dim <= 0 for dim in weight.shape):
                return False

        return True
