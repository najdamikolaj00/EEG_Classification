from torch import nn


class SequenceTrainer(nn.Module):
    def __init__(
        self,
        models: dict,
        recon,
        ncritic,
        losses_list,
        epochs,
        retain_checkpoints,
        checkpoints,
        mlflow_interval,
        device,
    ):
        super().__init__()
