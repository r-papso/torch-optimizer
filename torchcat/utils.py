from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def evaluate(model: nn.Module, data: Iterable[Tuple[Tensor, Tensor]]) -> float:
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in data:
            if inputs.device != device:
                inputs = inputs.to(device)

            if labels.device != device:
                labels = labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total
