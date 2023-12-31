import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from LSTM import LSTMModelBinary
from BCIDataset import BCIDatasetBinary

def train_val_test(device, model, num_epochs, num_splits, batch_size, learning_rate):
    train_dataset = BCIDatasetBinary(data_paths=["Data/A01T.mat"])

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"FOLD {fold + 1}")
        print("--------------------------------")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # Define data loaders for training and validation data in this fold
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                sampler=val_subsampler)

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy loss

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}")
            current_loss = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                targets = targets.to(device)
                data = data.to(device)

                optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs.float().squeeze(), targets.float())

                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                    )
                    current_loss = 0

        print("Training process has finished.")
        print("Starting validation")

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for idx, (data, targets) in enumerate(val_loader):
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

        print(f"Results for fold {fold + 1}:")
        print("--------------------------------")

        print(f"Validation Accuracy: {accuracy:.2f}")
        print(f"Validation Precision: {precision:.2f}")
        print(f"Validation Recall: {recall:.2f}")
        print(f"Validation F1 Score: {f1:.2f}")

    # Test loop
    test_dataset = BCIDatasetBinary(data_paths=["Data/A01E.mat"])
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

    print("Test Results:")
    print("-----------------")
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    print(f"Test F1 Score: {f1:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters, loop
    num_epochs = 50
    batch_size = 64
    num_splits = 5

    # Hyperparameters, model
    input_size = 1125  # Number of time points
    num_channels = 22  # Number of EEG channels
    hidden_size = 64
    num_layers = 4

    num_classes = 2
    lr = 0.001

    model = LSTMModelBinary(input_size, hidden_size, num_layers)

    train_val_test(device, model, num_epochs, num_splits, batch_size, lr)
