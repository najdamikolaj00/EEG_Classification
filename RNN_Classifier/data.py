import torch
from torch.utils.data import Dataset
from torchgan.models import Generator


def synthesis_df(
    generator: Generator, dataset: Dataset
) -> tuple[torch.Tensor, torch.Tensor]:
    pass
