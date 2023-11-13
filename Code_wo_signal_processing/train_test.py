import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from LSTM import LSTMModel
from CNN_LSTM import CNNLSTMModel
from BCIDataset import BCIDataset

def train_test(device, model, num_epochs, batch_size, learning_rate):
    train_dataset = BCIDataset(data_paths=["Data/A01T.mat"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        current_loss = 0

        for batch_idx, (data, targets) in enumerate(train_dataloader):
            targets = targets.to(device)
            data = data.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs.float(), targets.long())

            loss.backward()
            optimizer.step()

            current_loss += loss.item()

            if (batch_idx + 1) % 2 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                current_loss = 0

    print("Training process has finished.")

    # Test loop
    test_dataset = BCIDataset(data_paths=["Data/A01E.mat"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            targets = targets.to(device)
            data = data.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    print(f"Test F1 Score: {f1:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters, loop
    num_epochs = 1000
    batch_size = 64

    # Hyperparameters, model
    input_size = 1125  # Number of time points
    num_channels = 22  # Number of EEG channels
    hidden_size = 64
    num_layers = 2

    num_classes = 4
    lr = 0.0001

    # Specify the kernel size for the convolutional layer
    kernel_size = 3

    # Instantiate the model
    model = CNNLSTMModel(num_channels, hidden_size, num_layers, num_classes, kernel_size)

    #model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.to(device)

    train_test(device, model, num_epochs, batch_size, lr)
