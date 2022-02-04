from typing import Iterable, Tuple
from torch import nn

from ..prune import apply_mask


def dim_mask(module: nn.Module, name: str, dim: int) -> Iterable[bool]:
    if not hasattr(module, f"{name}_mask"):
        return [True] * module.weight.shape[dim]

    ndim = getattr(module, f"{name}_mask").ndim
    dims_to_sum = tuple(range(dim + 1, ndim))
    mask_sum = module.weight_mask.sum(dim=dims_to_sum)

    return mask_sum != 0


def reduce_parameter(module: nn.Module, name: str, slices: Tuple[slice]) -> None:
    param = module._parameters.get(name)
    if param is not None:
        param.data = param.data[slices]
        param.grad = None

    param_mask = module._buffers.get(f"{name}_mask")
    if param_mask is not None:
        module._buffers[f"{name}_mask"] = param_mask[slices]

    param_orig = module._parameters.get(f"{name}_orig")
    if param_orig is not None:
        param_orig.data = param_orig.data[slices]
        param_orig.grad = None

    if param_mask is not None and param_orig is not None:
        setattr(module, name, apply_mask(module, name))


def create_mask_slices(
    module: nn.Module, name: str, dim_mask: Iterable[bool], dim: int
) -> Tuple[slice, ...]:
    param_ndim = getattr(module, name).ndim
    return tuple([tuple(dim_mask) if i == dim else slice(None) for i in range(param_ndim)])


def set_out_shape(module: nn.Module, changed_dim: int, new_shape: int) -> None:
    out_shape = module._buffers.get("out_shape")
    if out_shape is not None:
        out_shape[changed_dim] = new_shape
        module._buffers["out_shape"] = out_shape


def module_ndim(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return 2
    elif isinstance(
        module, (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d)
    ):
        return 3
    elif isinstance(
        module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)
    ):
        return 4
    elif isinstance(
        module, (nn.Conv3d, nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)
    ):
        return 5
    else:
        return -1
