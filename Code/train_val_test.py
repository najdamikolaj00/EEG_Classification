from itertools import compress
from pathlib import Path
from typing import Sequence

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Code.BCIDataset import BCIDataset


def train_val_test(
    device,
    model: nn.Module,
    num_epochs: int,
    num_splits: int,
    batch_size: int,
    lr: float,
    subjects: Sequence = None,
):
    if subjects is None:
        subjects = (1, 1, 0, 0, 0, 0, 0, 0, 0)
    data_files = tuple(Path("Data").iterdir())
    if len(subjects) != len(data_files) // 2:
        raise ValueError(
            f"Subjects should be sequence of length equal to number of data files of a type. Should be {len(data_files) // 2} is {len(subjects)}"
        )
    train_data_paths = list(
        compress(sorted(path for path in data_files if "T" in path.name), subjects)
    )
    test_data_paths = list(
        compress(sorted(path for path in data_files if "E" in path.name), subjects)
    )
    train_dataset = BCIDataset(data_paths=train_data_paths)
    # print(train_dataset.__len__())
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # for idx, (data, label) in enumerate(train_dataloader):
    #     print(data)
    #     print(label)
    #     print(data.shape)
    #     print(label.shape)
    #     print(data.dtype)
    #     print(label.dtype)
    #     if idx == 0:
    #         break
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"FOLD {fold + 1}")
        print("--------------------------------")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # Define data loaders for training and validation data in this fold
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_subsampler
        )
        val_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=val_subsampler
        )

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}")
            current_loss = 0

            for batch_idx, (data, targets) in tqdm(
                enumerate(train_loader),
                desc=f"Running epoch {epoch + 1}",
                total=len(train_loader),
            ):
                targets = targets.to(device)
                data = data.to(device)

                optimizer.zero_grad()

                outputs = model(data)
                # print(targets.shape, outputs.shape)
                loss = criterion(outputs.float(), targets.long())

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

        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Precision: {precision:.2f}")
        print(f"Validation Recall: {recall:.2f}")
        print(f"Validation F1 Score: {f1:.2f}")

    # Test loop
    test_dataset = BCIDataset(data_paths=test_data_paths)
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

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    print(f"Test F1 Score: {f1:.2f}")
