from typing import List
from torch import nn
import torch
from pruning.layer.abstract import LayerPruner


class BatchNorm2dPruner(LayerPruner):
    _running_mean_name = "running_mean"
    _running_var_name = "running_var"

    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module) -> List[int]:
        raise ValueError(f"{BatchNorm2dPruner} -> Pruning by mask is not supported.")

    def prune_by_indicies(self, layer: nn.Module, indicies: List[int]) -> None:
        assert isinstance(
            layer, nn.BatchNorm2d
        ), f"{BatchNorm2dPruner} -> Layer must be instance of {nn.BatchNorm2d}."

        assert all(
            [idx >= 0 and idx < layer.weight.size().numel() for idx in indicies]
        ), f"{BatchNorm2dPruner} -> Indicies out of bounds."

        layer.zero_grad()

        with torch.no_grad():
            mask = torch.ones_like(layer.weight, dtype=torch.bool)
            mask[indicies] = False

            # Prune running_mean and running_var if present in the layer
            if (mean := layer._buffers.get("running_mean")) is not None:
                layer._buffers["running_mean"] = mean[mask]

            if (var := layer._buffers.get("running_var")) is not None:
                layer._buffers["running_var"] = var[mask]

            # Prune weight and bias parameters if present in the layer
            if (weight := layer._parameters.get("weight")) is not None:
                weight.data = weight.data[mask]

            if (bias := layer._parameters.get("bias")) is not None:
                bias.data = weight.data[mask]

            # Prune weight and bias masks and orig parameters if present in the layer
            if (weight_mask := layer._buffers.get("weight_mask")) is not None:
                layer._buffers["weight_mask"] = weight_mask[mask]

            if (weight_orig := layer._parameters.get("weight_orig")) is not None:
                weight_orig.data = weight_orig.data[mask]

            if (bias_mask := layer._buffers.get("bias_mask")) is not None:
                layer._buffers["bias_mask"] = bias_mask[mask]

            if (bias_orig := layer._parameters.get("bias_orig")) is not None:
                bias_orig.data = bias_orig.data[mask]
