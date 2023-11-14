from dataclasses import dataclass

import numpy as np

from pre_processing.reformat_input import reformat_input


@dataclass
class CSP:
    classes: tuple[str, str]
    algorithm: np.array

    def apply(self, data: np.array) -> tuple[float, float]:
        trails = reformat_input(data)
        n_channels, n_samples, n_trials = trails.shape
        trails_csp = np.zeros_like(trails)
        for i in range(n_trials):
            trails_csp[:, :, i] = self.algorithm.T.dot(trails[:, :, i])
        return np.log(np.var(trails_csp[[0, -1], :, 0], axis=1))
