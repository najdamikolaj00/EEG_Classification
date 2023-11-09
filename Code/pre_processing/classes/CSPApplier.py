from dataclasses import dataclass
from itertools import chain

from torch import Tensor

from Code.pre_processing.classes.CSP import CSP


@dataclass
class CSPApplier:
    csps: list[CSP]

    def apply(self, data: Tensor):
        return tuple(chain.from_iterable(csp.apply(data) for csp in self.csps))
