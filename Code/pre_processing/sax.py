from typing import NamedTuple, Sequence, Self

from numpy import mean, corrcoef
from torch import Tensor


class Sax(NamedTuple):
    mean: float
    coefficient: float

    @classmethod
    def from_sequence(cls, segment: Sequence) -> Self:
        segment = tuple(segment)
        return Sax(mean(segment), corrcoef(range(len(segment)), segment)[0, 1])


def get_sax(
    data: Tensor, segment_length: int
) -> tuple[tuple[tuple[Sax, ...], ...], ...]:
    return tuple(
        tuple(calc_sax(channel, segment_length) for channel in sample[0])
        for sample in data
    )


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
