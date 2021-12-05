import numpy as np

from dataclasses import dataclass


@dataclass
class LayerMask:
    name: str
    mask: np.ndarray
