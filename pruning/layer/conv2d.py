from typing import List
from torch import nn
import torch
from pruning.layer.abstract import LayerPruner


class Conv2dPruner(LayerPruner):
    def __init__(self) -> None:
        super().__init__()

    def prune_by_mask(self, layer: nn.Module) -> List[int]:
        assert isinstance(
            layer, nn.Conv2d
        ), f"{Conv2dPruner} -> Layer must be instance of {nn.Conv2d}."

        assert hasattr(layer, "weight_mask"), f"{Conv2dPruner} -> Weight mask in layer not set."

        layer.zero_grad()

        with torch.no_grad():
            axes_to_sum = tuple(range(1, len(layer.weight_mask.shape)))
            mask_sum = layer.weight_mask.sum(axes_to_sum)
            list_idxs = (mask_sum == 0).nonzero().squeeze().tolist()

            w_mask = torch.ones_like(layer.weight, dtype=torch.bool)
            w_mask[list_idxs] = False
            b_mask = torch.ones_like(layer.bias, dtype=torch.bool)
            b_mask[list_idxs] = False

            # Prune weight and bias parameters
            weight = layer._parameters.get("weight")
            weight.data = weight.data[w_mask]

            bias = layer._parameters.get("bias")
            bias.data = weight.data[b_mask]

            # Prune weight and bias masks and orig parameters if present in the layer
            if (weight_mask := layer._buffers.get("weight_mask")) is not None:
                layer._buffers["weight_mask"] = weight_mask[w_mask]

            if (weight_orig := layer._parameters.get("weight_orig")) is not None:
                weight_orig.data = weight_orig.data[w_mask]

            if (bias_mask := layer._buffers.get("bias_mask")) is not None:
                layer._buffers["bias_mask"] = bias_mask[b_mask]

            if (bias_orig := layer._parameters.get("bias_orig")) is not None:
                bias_orig.data = bias_orig.data[b_mask]

    def prune_by_indicies(self, layer: nn.Module, indicies: List[int]) -> None:
        assert isinstance(
            layer, nn.Conv2d
        ), f"{Conv2dPruner} -> Layer must be instance of {nn.Conv2d}."

        layer.zero_grad()

        with torch.no_grad():
            w_mask = torch.ones_like(layer.weight_mask, dtype=torch.bool)
            w_mask[:, indicies, :, :] = False

            # Prune weight parameter
            weight = layer._parameters.get("weight")
            weight.data = weight.data[w_mask]

            # Prune weight mask and orig parameter if present in the layer
            if (weight_mask := layer._buffers.get("weight_mask")) is not None:
                layer._buffers["weight_mask"] = weight_mask[w_mask]

            if (weight_orig := layer._parameters.get("weight_orig")) is not None:
                weight_orig.data = weight_orig.data[w_mask]
