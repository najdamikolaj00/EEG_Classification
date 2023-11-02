import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import BCIDataset
from LSTM import LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1125  # Number of time points
num_channels = 22  # Number of EEG channels
batch_size = 2
num_classes = 4

train_dataset = BCIDataset(data_path='Data/A01T.mat')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# testing data shapes
# for batch_idx, (data, targets) in enumerate(train_loader):
#     print('Data:', data.shape)
#     print('Targets:', targets.shape)

model = LSTMModel(input_size=num_channels, hidden_size=64, num_layers=2, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
  
        loss = criterion(outputs, targets)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{2}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{2}], Average Loss: {average_loss:.4f}')

#model = ATCNet().to(device)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0

#     for batch_idx, (data, targets) in enumerate(train_loader):
#         data, targets = data.to(device), targets.to(device)

#         outputs = model(data)

#         loss = criterion(outputs, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         if batch_idx % 100 == 0:
#             print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

#     average_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

