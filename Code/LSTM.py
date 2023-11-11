import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dtype=torch.float64
        )
        self.fc = nn.Linear(hidden_size, num_classes).to(torch.float64)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64
        ).to(x.device)

        batch_size, dimension, num_channels, time_points = x.size()
        x = x.view(batch_size, time_points, -1)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, time_points, num_channels*n_dim)

        out = self.fc(out[:, -1, :])
        return out
