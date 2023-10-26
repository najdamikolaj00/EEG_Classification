from torch.utils.data import DataLoader

from Code.Dataset import BCIDataset
from Code.ATCNet_our_approach import ATCNet

if __name__ == "__main__":
    dataset = BCIDataset("Data/A01E.mat")
    data_loader = DataLoader(dataset, batch_size=2)
    net = ATCNet()

    for batch_index, (sample, target) in enumerate(data_loader):
        prediction = net(sample)
        pass
