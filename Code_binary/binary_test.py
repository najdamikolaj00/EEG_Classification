from torch.utils.data import DataLoader
from BCIDataset import BCIDatasetBinary

data_paths = ["Data/A01T.mat"]

dataset = BCIDatasetBinary(data_paths, target_classes=[1, 2])
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_idx, (data, targets) in enumerate(dataloader):
    # Check the distribution of classes in the batch
    class_distribution = {class_label: (targets == class_label).sum().item() for class_label in set(targets.numpy())}
    print(f"Batch [{batch_idx + 1}], Class Distribution: {class_distribution}")
