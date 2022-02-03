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
    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        pass

    @abstractmethod
    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        pass


class ReducerBase(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        self._before_reduce(module)

        with torch.no_grad():
            first_dim_mask = utils.first_dim_mask(module, "weight")

            w_slices = utils.create_mask_slices(module, "weight", first_dim_mask, 0)
            utils.reduce_parameter(module, "weight", w_slices)

            if getattr(module, "bias", None) is not None:
                b_slices = utils.create_mask_slices(module, "bias", first_dim_mask, 0)
                utils.reduce_parameter(module, "bias", b_slices)

        self._after_reduce(module, first_dim_mask)
        return self._get_reduced_dims(module, first_dim_mask)

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        self._before_adjust(module, reduced_dims)

        with torch.no_grad():
            slices = utils.create_mask_slices(module, "weight", reduced_dims[1], 1)
            utils.reduce_parameter(module, "weight", slices)

        self._after_adjust(module, reduced_dims)
        return None

    @abstractmethod
    def _get_reduced_dims(
        self, module: nn.Module, reduced_dim: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        pass

    def _before_reduce(self, module: nn.Module) -> None:
        pass

    def _before_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        pass

    def _after_reduce(self, module: nn.Module, reduced_dim: Iterable[bool]) -> None:
        pass

    def _after_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        pass


class ConvReducer(ReducerBase):
    _conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

    def __init__(self) -> None:
        super().__init__()

    def _get_reduced_dims(
        self, module: nn.Module, reduced_dim: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        last_dims = self.__last_out_dims(module)
        return (None, reduced_dim) + last_dims

    def _before_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, self._conv_types), f"Invalid module type: {type(module)}."

    def _before_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, self._conv_types), f"Invalid module type: {type(module)}."

    def _after_reduce(self, module: nn.Module, reduced_dim: Iterable[bool]) -> None:
        setattr(module, "out_channels", sum(reduced_dim))
        utils.set_out_shape(module, 1, sum(reduced_dim))

    def _after_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_channels", sum(reduced_dims[1]))

    def __last_out_dims(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        out_shape = getattr(module, "out_shape", None)
        if out_shape is not None:
            dim_masks = tuple([[True] * out_shape[i] for i in range(2, out_shape.ndim)])
        else:
            dim_masks = tuple([None] * (utils.module_ndim(module) - 2))

        return dim_masks


class LinearReducer(ReducerBase):
    def __init__(self) -> None:
        super().__init__()

    def _get_reduced_dims(
        self, module: nn.Module, reduced_dim: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        out_shape = getattr(module, "out_shape", None)
        if out_shape is not None:
            out = [[True] * out_dim for out_dim in out_shape.tolist()]
            out[-1] = reduced_dim
        else:
            out = [None, reduced_dim]

        return tuple(out)

    def _before_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, nn.Linear), f"Invalid module type: {type(module)}."

    def _before_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, nn.Linear), f"Invalid module type: {type(module)}."

    def _after_reduce(self, module: nn.Module, reduced_dim: Iterable[bool]) -> None:
        setattr(module, "out_features", sum(reduced_dim))
        utils.set_out_shape(module, -1, sum(reduced_dim))

    def _after_adjust(self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_features", sum(reduced_dims[-1]))


class BatchNormReducer(Reducer):
    _batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._batchnorm_types), f"Invalid module type: {type(module)}."

        with torch.no_grad():
            slices = (reduced_dims[1],)
            param_names = ["running_mean", "running_var", "weight", "bias"]

            for name in param_names:
                utils.reduce_parameter(module, name, slices)

        module.num_features = sum(reduced_dims[1])
        utils.set_out_shape(module, 1, sum(reduced_dims[1]))

        return reduced_dims


class FlattenReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, nn.Flatten), f"Invalid module type: {type(module)}."

        start = module.start_dim
        end = module.end_dim if module.end_dim != -1 else len(reduced_dims)
        dim_masks = reduced_dims[start:end]

        assert all(mask is not None for mask in dim_masks)

        flattened = [all(vals) for vals in itertools.product(*dim_masks)]
        utils.set_out_shape(module, start, sum(flattened))

        return reduced_dims[:start] + (flattened,) + reduced_dims[end:]


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

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._pool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(reduced_dims[1]))
        return (reduced_dims[0], reduced_dims[1]) + out_dims_masks

    def __out_dims_masks(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        out_shape = getattr(module, "out_shape", None)
        if out_shape is not None:
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

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._adaptivepool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(reduced_dims[1]))
        return (reduced_dims[0], reduced_dims[1]) + out_dims_masks

    def __out_dims_masks(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        ndim = utils.module_ndim(module)
        out_sizes = (
            module.output_size
            if isinstance(module.output_size, tuple)
            else (module.output_size,) * (ndim - 2)
        )

        out_shape = getattr(module, "out_shape", None)
        if out_shape is not None:
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

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def adjust(
        self, module: nn.Module, reduced_dims: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        for i, dim_mask in enumerate([d for d in reduced_dims if d is not None]):
            utils.set_out_shape(module, i, sum(dim_mask))

        return reduced_dims
