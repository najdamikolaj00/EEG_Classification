import numpy as np


def reformat_input(data: np.array) -> np.array:
    """
    Reformats data to make it usable in csp algorith
    :param data: (trails x channels x samples)
    :return: (channels x samples x trials)
    """
    return np.transpose(data, (1, 2, 0))
