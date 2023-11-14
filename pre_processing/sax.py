from typing import NamedTuple, Sequence, Self

import numpy as np
from numpy import mean, corrcoef
from torch import Tensor


class SaxArgs(NamedTuple):
    segment_length: int
    sax_only: bool = True


class Sax(NamedTuple):
    mean: float
    coefficient: float

    @classmethod
    def from_sequence(cls, segment: Sequence) -> Self:
        segment = tuple(segment)
        return Sax(mean(segment), corrcoef(range(len(segment)), segment)[0, 1])


def get_sax(data: Tensor, segment_length: int) -> np.array:
    return conv_to_array(
        tuple(
            tuple(calc_sax(channel, segment_length) for channel in sample[0])
            for sample in data
        )
    )


def conv_to_array(saxs: tuple[tuple[tuple[Sax, ...], ...], ...]) -> np.array:
    return np.array(
        tuple(
            tuple(np.reshape(channel, -1) for channel in trail)
            for trail in np.array(saxs)
        )
    )[:, np.newaxis, :, :]


def calc_sax(sequence: Sequence, segment_length: int) -> tuple[Sax, ...]:
    return tuple(
        map(
            Sax.from_sequence,
            (
                sequence[i : i + segment_length]
                for i in range(0, len(sequence), segment_length)
            ),
        )
    )
