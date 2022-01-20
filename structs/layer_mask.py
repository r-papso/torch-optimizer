import numpy as np

from dataclasses import dataclass
from torch import nn


@dataclass
class LayerMask:
    name: str
    layer: nn.Module
    mask: np.ndarray
