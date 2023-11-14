import numpy as np


def isoelectric_line_removal(data_array: np.array, samples_last=True):
    if not samples_last:
        raise ValueError("Not implemented yet")
    return data_array - np.mean(data_array, axis=-1)[:, :, :, np.newaxis]
