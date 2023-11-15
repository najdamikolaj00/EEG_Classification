import torch
import torch.nn as nn

class LSTMModelBinary(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModelBinary, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dtype=torch.float64)
        self.fc = nn.Linear(hidden_size, 1).to(torch.float64)  # Change num_classes to 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), 
                         self.hidden_size, dtype=torch.float64).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), 
                         self.hidden_size, dtype=torch.float64).to(x.device)

        batch_size, dimension, num_channels, time_points = x.size()
        x = x.view(batch_size, num_channels*dimension, time_points)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # Select the output at the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)  # Apply sigmoid activation for binary classification
        return out
