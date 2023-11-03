import torch
from torch import nn
from torch.utils.data import DataLoader

from Code.ATCNet.Dataset import BCIDataset
from Code.ATCNet.ATCNet_our_approach import ATCNet

dataset = BCIDataset("Data/A01E.mat")
train_loader = DataLoader(dataset, batch_size=10)
model = ATCNet(eegn_poolSize=7)
model = model.float()
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
            print(
                f"Epoch [{epoch + 1}/{2}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{2}], Average Loss: {average_loss:.4f}")
