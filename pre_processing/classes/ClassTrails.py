from dataclasses import dataclass

import numpy as np


@dataclass
class ClassTrails:
    class_: int
    data: np.array
