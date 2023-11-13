import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from LSTM import LSTMModel
from CNN_LSTM import CNNLSTMModel
from BCIDataset import BCIDataset

def train_test_loso(device, model, num_epochs, batch_size, learning_rate):
    subject_data_paths = ["Data/A01T.mat", "Data/A02T.mat", "Data/A03T.mat", 
                          "Data/A04T.mat", "Data/A05T.mat", "Data/A06T.mat",
                          "Data/A07T.mat", "Data/A08T.mat", "Data/A09T.mat"]  # Add paths for all subjects

    for test_subject_path in subject_data_paths:
        print(f"\nLeaving out subject: {test_subject_path}")

        # Train on all other subjects
        train_subjects = [path for path in subject_data_paths if path != test_subject_path]
        train_dataset = BCIDataset(data_paths=train_subjects)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()  # Set the model to training mode

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch + 1}")
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

        # Test on the left-out subject
        test_dataset = BCIDataset(data_paths=[test_subject_path])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model.eval()  # Set the model to evaluation mode

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

        print(f"Test Accuracy for {test_subject_path}: {accuracy:.2f}")
        print(f"Test Precision for {test_subject_path}: {precision:.2f}")
        print(f"Test Recall for {test_subject_path}: {recall:.2f}")
        print(f"Test F1 Score for {test_subject_path}: {f1:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters, loop
    num_epochs = 20
    batch_size = 64

    # Hyperparameters, model
    input_size = 1125  # Number of time points
    num_channels = 22  # Number of EEG channels
    hidden_size = 64
    num_layers = 3

    num_classes = 4
    lr = 0.001

    # Specify the kernel size for the convolutional layer
    kernel_size = 3

    # Instantiate the model
    model = CNNLSTMModel(num_channels, hidden_size, num_layers, num_classes, kernel_size)

    #model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.to(device)

    train_test_loso(device, model, num_epochs, batch_size, lr)
