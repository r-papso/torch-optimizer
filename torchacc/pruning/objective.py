import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader

warnings.simplefilter("ignore", UserWarning)


class Objective(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        pass


class ObjectiveContainer(Objective):
    def __init__(self, objectives: Iterable[Objective]) -> None:
        super().__init__()

        self._objs = objectives

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        obj_vals = [obj.evaluate(model) for obj in self._objs]
        max_len = max(len(obj_val) for obj_val in obj_vals)
        result = [0.0] * max_len

        for obj_val in obj_vals:
            for i in range(max_len):
                result[i] += obj_val[i] if i < len(obj_val) else 0.0

        return tuple(result)


class Accuracy(Objective):
    def __init__(self, val_loader: DataLoader, device: str) -> None:
        super().__init__()

        self._loader = val_loader
        self._device = device

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in self._loader:
                inputs = inputs.to(self._device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        return (correct / total,)


class MacsPenalty(Objective):
    def __init__(self, weight: float, p: float, orig_macs: int, in_shape: Tuple[int, ...],) -> None:
        super().__init__()

        self._weigh = weight
        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = in_shape

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        macs, _ = profile(model, inputs=(torch.randn(self._input_shape),), verbose=False)
        penalty = self._weigh * max(0.0, macs - self._orig_macs * self._p)
        return (penalty,)
