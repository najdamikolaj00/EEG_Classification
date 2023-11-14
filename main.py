import torch

from models.ATCNet import ATCNet
from Code.LSTM import LSTMModel
from Code.train_val_test import train_val_test
from constants import ModelType

if __name__ == "__main__":
    model_type = ModelType.ATC_NET
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters, loop
    num_epochs = 10
    batch_size = 2
    num_splits = 5

    # Hyperparameters, model
    input_size = 1125  # Number of time points
    num_channels = 22  # Number of EEG channels
    hidden_size = 64
    num_layers = 2

    num_classes = 4
    lr = 0.001
    if model_type == ModelType.LSTM:
        model = LSTMModel(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    elif model_type == ModelType.ATC_NET:
        model = ATCNet(eegn_poolSize=7, n_classes=num_classes)
    else:
        raise ValueError(f"No such {model_type=}")
    train_val_test(device, model, num_epochs, num_splits, batch_size, lr)
