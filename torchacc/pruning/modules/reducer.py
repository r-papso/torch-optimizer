import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
from torch import nn

from . import utils


class Reducer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        pass

    @abstractmethod
    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        pass

    # TODO: implement
    def reduce_dim(
        self, module: nn.Module, dim_mask: Iterable[bool], dim: int
    ) -> Tuple[Iterable[bool], ...]:
        pass


class ReducerBase(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        self._before_mask_reduce(module)
        module.zero_grad()

        with torch.no_grad():
            channel_mask = utils.channel_mask(module, "weight")

            w_slices = utils.create_slices(module, "weight", channel_mask, 0)
            utils.reduce_parameter(module, "weight", w_slices)

            if getattr(module, "bias", None) is not None:
                b_slices = utils.create_slices(module, "bias", channel_mask, 0)
                utils.reduce_parameter(module, "bias", b_slices)

        self._after_mask_reduce(module, channel_mask)
        return self._mask_reduce_result(module, channel_mask)

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        self._befor_input_reduce(module, input)
        module.zero_grad()

        with torch.no_grad():
            dim = self._input_dependent_dim()
            slices = utils.create_slices(module, "weight", input[dim], dim)
            utils.reduce_parameter(module, "weight", slices)

        self._after_input_reduce(module, input)
        return self._input_reduce_result(module, input)

    @abstractmethod
    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        pass

    @abstractmethod
    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[int], ...]
    ) -> Tuple[Iterable[int], ...]:
        pass

    @abstractmethod
    def _input_dependent_dim(self) -> int:
        pass

    def _before_mask_reduce(self, module: nn.Module) -> None:
        pass

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        pass

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        pass

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        pass


class ConvReducer(ReducerBase):
    _conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

    def __init__(self) -> None:
        super().__init__()

    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        out_dims_masks = self.__out_dims_masks(module)
        return (None, channel_mask) + out_dims_masks

    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        return None

    def _input_dependent_dim(self) -> int:
        return 1

    def _before_mask_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, self._conv_types), f"Invalid module type: {type(module)}."

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, self._conv_types), f"Invalid module type: {type(module)}."

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        setattr(module, "out_channels", sum(channel_mask))
        utils.set_out_shape(module, 1, sum(channel_mask))

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_channels", sum(input[1]))

    def __out_dims_masks(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        if (out_shape := getattr(module, "out_shape", None)) is not None:
            dim_masks = tuple([[True] * out_shape[i] for i in range(2, out_shape.ndim)])
        else:
            dim_masks = tuple([None] * (utils.module_ndim(module) - 2))

        return dim_masks


class LinearReducer(ReducerBase):
    def __init__(self) -> None:
        super().__init__()

    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        if (out_shape := getattr(module, "out_shape", None)) is not None:
            out = [[True] * out_dim for out_dim in out_shape.tolist()]
            out[-1] = channel_mask
        else:
            out = [None, channel_mask]

        return tuple(out)

    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        return None

    def _input_dependent_dim(self) -> int:
        return -1

    def _before_mask_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, nn.Linear), f"Invalid module type: {type(module)}."

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, nn.Linear), f"Invalid module type: {type(module)}."

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        setattr(module, "out_features", sum(channel_mask))
        utils.set_out_shape(module, -1, sum(channel_mask))

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_features", sum(input[-1]))


class BatchNormReducer(Reducer):
    _batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._batchnorm_types), f"Invalid module type: {type(module)}."

        module.zero_grad()
        with torch.no_grad():
            slices = (input[1],)
            param_names = ["running_mean", "running_var", "weight", "bias"]

            for name in param_names:
                utils.reduce_parameter(module, name, slices)

        module.num_features = sum(input[1])
        utils.set_out_shape(module, 1, sum(input[1]))

        return input


class FlattenReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, nn.Flatten), f"Invalid module type: {type(module)}."

        start = module.start_dim
        end = module.end_dim if module.end_dim != -1 else len(input)
        dim_masks = input[start:end]

        assert all(mask is not None for mask in dim_masks)

        flattened = [all(vals) for vals in itertools.product(*dim_masks)]
        utils.set_out_shape(module, start, sum(flattened))

        return input[:start] + (flattened,) + input[end:]


class PoolReducer(Reducer):
    _pool_types = (
        nn.MaxPool1d,
        nn.AvgPool1d,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.MaxPool3d,
        nn.AvgPool3d,
    )

    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._pool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(input[1]))
        return (input[0], input[1]) + out_dims_masks

    def __out_dims_masks(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        if (out_shape := getattr(module, "out_shape", None)) is not None:
            dim_masks = tuple([[True] * out_shape[i] for i in range(2, out_shape.ndim)])
        else:
            dim_masks = tuple([None] * (utils.module_ndim(module) - 2))

        return dim_masks


class AdaptivePoolReducer(Reducer):
    _adaptivepool_types = (
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool3d,
    )

    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._adaptivepool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(input[1]))
        return (input[0], input[1]) + out_dims_masks

    def __out_dims_masks(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        ndim = utils.module_ndim(module)
        out_sizes = (
            module.output_size
            if isinstance(module.output_size, tuple)
            else (module.output_size,) * (ndim - 2)
        )

        if (out_shape := getattr(module, "out_shape", None)) is not None:
            out_buff = tuple([True] * out_shape[i] for i in range(2, ndim))
        else:
            out_buff = tuple([None] * (ndim - 2))

        out = tuple(
            [True] * out_sizes[i] if out_sizes[i] is not None else out_buff[i]
            for i in range(ndim - 2)
        )

        return out


class FixedOpReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        for i, dim_mask in enumerate([d for d in input if d is not None]):
            utils.set_out_shape(module, i, sum(dim_mask))

        return input
