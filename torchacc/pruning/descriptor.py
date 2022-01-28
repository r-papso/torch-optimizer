from typing import Tuple
from torch import nn
import torch


def set_dims(model: nn.Module, input: Tuple[int, ...]) -> None:
    x = torch.rand((1,) + input)

    for _, module in model.named_modules():
        module.register_forward_hook(_DescriptorHook())

    y = model.forward(x)

    for _, module in model.named_modules():
        del_key = next(
            k for k, hook in module._forward_hooks.items() if isinstance(hook, _DescriptorHook)
        )
        del module._forward_hooks[del_key]


class _DescriptorHook:
    def __init__(self) -> None:
        pass

    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        out_shape = list(output.data.size())
        out_shape[0] = 0
        module.register_buffer("out_shape", torch.tensor(out_shape))
