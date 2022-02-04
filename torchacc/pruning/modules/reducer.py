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
    def reduce_dim(self, module: nn.Module, name: str, dim: int, dim_mask: Iterable[bool]) -> None:
        pass

    @abstractmethod
    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        pass


class ReducerBase(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        zero_dim_mask = utils.dim_mask(module, "weight", 0)
        _ = self.reduce_dim(module, "bias", 0, zero_dim_mask)
        return self.reduce_dim(module, "weight", 0, zero_dim_mask)

    def reduce_dim(
        self, module: nn.Module, name: str, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        assert 0 <= dim and dim < getattr(module, name).ndim

        self._before_reduce_dim(module, name, dim, dim_mask)

        with torch.no_grad():
            if getattr(module, name, None) is not None:
                slices = utils.create_mask_slices(module, name, dim_mask, dim)
                utils.reduce_parameter(module, name, slices)

        self._after_reduce_dim(module, name, dim, dim_mask)
        return self._get_output_dim_masks(module, name, dim, dim_mask)

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        in_dim = self._in_dependent_dim()
        return self.reduce_dim(module, "weight", in_dim, dim_masks[1])

    def _get_output_dim_masks(
        self, module: nn.Module, name: str, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        out_shape = getattr(module, "out_shape", None)

        if out_shape is not None:
            out = [[True] * out_dim if out_dim > 0 else None for out_dim in out_shape.tolist()]
        else:
            out = [None] * (utils.module_ndim(module))

        if name == "weight" and dim == 0:
            out[self._out_dependent_dim()] = dim_mask

        return tuple(out)

    def _before_reduce_dim(
        self, module: nn.Module, name: str, dim: int, dim_mask: Iterable[bool]
    ) -> None:
        assert isinstance(module, self._allowed_types()), f"Invalid module type: {type(module)}."

    def _after_reduce_dim(
        self, module: nn.Module, name: str, dim: int, dim_mask: Iterable[bool]
    ) -> None:
        if name == "weight" and dim == 0:
            setattr(module, self._out_dim_property_name(), sum(dim_mask))
            utils.set_out_shape(module, self._out_dependent_dim(), sum(dim_mask))
        elif name == "weight" and dim == 1:
            setattr(module, self._in_dim_property_name(), sum(dim_mask))

    @abstractmethod
    def _in_dependent_dim(self) -> int:
        pass

    @abstractmethod
    def _out_dependent_dim(self) -> int:
        pass

    @abstractmethod
    def _in_dim_property_name(self) -> str:
        pass

    @abstractmethod
    def _out_dim_property_name(self) -> str:
        pass

    @abstractmethod
    def _allowed_types(self) -> Tuple[type]:
        pass


class ConvReducer(ReducerBase):
    _conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

    def __init__(self) -> None:
        super().__init__()

    def _in_dependent_dim(self) -> int:
        return 1

    def _out_dependent_dim(self) -> int:
        return 1

    def _in_dim_property_name(self) -> str:
        return "in_channels"

    def _out_dim_property_name(self) -> str:
        return "out_channels"

    def _allowed_types(self) -> Tuple[type]:
        return self._conv_types


class LinearReducer(ReducerBase):
    def __init__(self) -> None:
        super().__init__()

    def _in_dependent_dim(self) -> int:
        return 1

    def _out_dependent_dim(self) -> int:
        return -1

    def _in_dim_property_name(self) -> str:
        return "in_features"

    def _out_dim_property_name(self) -> str:
        return "out_features"

    def _allowed_types(self) -> Tuple[type]:
        return (nn.Linear,)


class BatchNormReducer(Reducer):
    _batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in BatchNorm is not supported.")

    def reduce_dim(
        self, module: nn.Module, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in BatchNorm is not supported.")

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._batchnorm_types), f"Invalid module type: {type(module)}."

        with torch.no_grad():
            slices = (dim_masks[1],)
            param_names = ["running_mean", "running_var", "weight", "bias"]

            for name in param_names:
                utils.reduce_parameter(module, name, slices)

        module.num_features = sum(dim_masks[1])
        utils.set_out_shape(module, 1, sum(dim_masks[1]))

        return dim_masks


class FlattenReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in Flatten is not supported.")

    def reduce_dim(
        self, module: nn.Module, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in Flatten is not supported.")

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, nn.Flatten), f"Invalid module type: {type(module)}."

        start = module.start_dim
        end = module.end_dim if module.end_dim != -1 else len(dim_masks)
        dim_masks = dim_masks[start:end]

        assert all(mask is not None for mask in dim_masks)

        flattened = [all(vals) for vals in itertools.product(*dim_masks)]
        utils.set_out_shape(module, start, sum(flattened))

        return dim_masks[:start] + (flattened,) + dim_masks[end:]


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
        raise ValueError(f"Reduction in Pool is not supported.")

    def reduce_dim(
        self, module: nn.Module, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in Pool is not supported.")

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._pool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(dim_masks[1]))
        return (dim_masks[0], dim_masks[1]) + out_dims_masks

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
        raise ValueError(f"Reduction in AdaptivePool is not supported.")

    def reduce_dim(
        self, module: nn.Module, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction in AdaptivePool is not supported.")

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, self._adaptivepool_types), f"Invalid module type: {type(module)}."

        out_dims_masks = self.__out_dims_masks(module)
        utils.set_out_shape(module, 1, sum(dim_masks[1]))
        return (dim_masks[0], dim_masks[1]) + out_dims_masks

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
        raise ValueError(f"Reduction is not supported.")

    def reduce_dim(
        self, module: nn.Module, dim: int, dim_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction is not supported.")

    def adjust(
        self, module: nn.Module, dim_masks: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        for i, dim_mask in enumerate([d for d in dim_masks if d is not None]):
            utils.set_out_shape(module, i, sum(dim_mask))

        return dim_masks
