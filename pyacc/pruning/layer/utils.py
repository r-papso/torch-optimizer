from typing import Iterable, Tuple
from torch import nn

from pyacc.pruning.prune import apply_mask


def channel_mask(module: nn.Module, name: str) -> Iterable[bool]:
    if not hasattr(module, f"{name}_mask"):
        return [True] * module.weight.shape[0]

    mask_shape = getattr(module, f"{name}_mask").shape
    dims_to_sum = tuple(range(1, len(mask_shape)))
    mask_sum = module.weight_mask.sum(dim=dims_to_sum)

    return mask_sum != 0


def zero_channels(module: nn.Module, name: str) -> Iterable[int]:
    if not hasattr(module, f"{name}_mask"):
        return []

    mask_shape = getattr(module, f"{name}_mask").shape
    dims_to_sum = tuple(range(1, len(mask_shape)))
    mask_sum = module.weight_mask.sum(dim=dims_to_sum)
    zero_idxs = (mask_sum == 0).nonzero().squeeze().tolist()

    return zero_idxs


def non_zero_channels(module: nn.Module, name: str) -> Iterable[int]:
    if not hasattr(module, f"{name}_mask"):
        return list(range(module.weight.shape[0]))

    mask_shape = getattr(module, f"{name}_mask").shape
    dims_to_sum = tuple(range(1, len(mask_shape)))
    mask_sum = module.weight_mask.sum(dim=dims_to_sum)
    non_zero_idxs = (mask_sum > 0).nonzero().squeeze().tolist()

    return non_zero_idxs


def reduce_parameter(module: nn.Module, name: str, slices: Tuple[slice]) -> None:
    if (param := module._parameters.get(name)) is not None:
        param.data = param.data[slices]

    if (param_mask := module._buffers.get(f"{name}_mask")) is not None:
        module._buffers[f"{name}_mask"] = param_mask[slices]

    if (param_orig := module._parameters.get(f"{name}_orig")) is not None:
        param_orig.data = param_orig.data[slices]

    if param_mask is not None and param_orig is not None:
        setattr(module, name, apply_mask(module, name))


def create_slices(
    module: nn.Module, name: str, dim_mask: Iterable[bool], dim: int
) -> Tuple[slice, ...]:
    param_ndim = getattr(module, name).ndim
    dim = dim if dim != -1 else param_ndim - 1
    return tuple([tuple(dim_mask) if i == dim else slice(None) for i in range(param_ndim)])


def set_out_shape(module: nn.Module, changed_dim: int, new_shape: int) -> None:
    if (out_shape := module._buffers.get("out_shape")) is not None:
        out_shape[changed_dim] = new_shape
        module._buffers["out_shape"] = out_shape
