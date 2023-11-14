import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size=3):
        super(CNNLSTMModel, self).__init__()

        # CNN layer
        self.conv1d = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # LSTM layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.5
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes).to(torch.float32)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32
        ).to(x.device)

        # Reshape input for the convolutional layer
        x = x.view(x.size(0), x.size(2), x.size(3))
        x = x.to(self.conv1d.weight.dtype)

        # Apply 1D convolution
        x = self.conv1d(x)
        x = self.batch_norm(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)

        # Dropout layer
        x = self.dropout(x)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Select the output at the last time step
        out = self.fc(out[:, -1, :])
        return out
