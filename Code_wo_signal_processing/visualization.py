import numpy as np
import matplotlib.pyplot as plt
from BCIDataset import BCIDataset
from torch.utils.data import DataLoader

def preprocess(data, start_time_point, end_time_point):
    # Reshape data to [batch_size, channels, time points]
    data = data.squeeze(1).numpy()

    # A subset of time points
    time_points_subset = slice(start_time_point, end_time_point)

    return data, time_points_subset

def visualize_single_eeg(data_path, start_time_point, end_time_point,
                        channel):
    
    dataset = BCIDataset(data_paths=[data_path])
    dataloader = DataLoader(dataset, batch_size=1)

    for data, label in dataloader:

        data, time_points_subset = preprocess(data, start_time_point, end_time_point)

        plt.figure(figsize=(12, 4))
        plt.style.use('bmh')
        plt.plot(data[0, channel, time_points_subset], label=f'Channel {channel}', color='black')
        plt.plot(np.zeros((end_time_point, 1)), '--', color='gray')
        plt.title(f'EEG Data - Channel {channel}')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

        break

def visualize_all_eeg(data_path, start_time_point, end_time_point, 
                      number_of_channels):
    dataset = BCIDataset(data_paths=[data_path])
    dataloader = DataLoader(dataset, batch_size=1)

    for data, label in dataloader:

        data, time_points_subset = preprocess(data, start_time_point, end_time_point)

        plt.figure(figsize=(10, 6))
        plt.style.use('bmh')
        plt.plot(data[0, :number_of_channels, time_points_subset].T 
                 + 30*np.arange(number_of_channels-1, -1, -1), color='black')
        plt.plot(np.zeros((end_time_point, number_of_channels)) 
                 + 30*np.arange(number_of_channels-1, -1, -1), '--', color='gray')
        plt.yticks([])
        plt.legend(range(1, number_of_channels), prop={'size': 10})
        plt.title('EEG Data - All Channels')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.show()

        break

if __name__ == '__main__':
    data_path = "Data/A01T.mat"

    # start_time_point = 0
    # end_time_point = 1125
    # number_of_channels = 22
    # visualize_all_eeg(data_path, start_time_point, end_time_point, 
    #                   number_of_channels)
    
    start_time_point = 0
    end_time_point = 1125
    channel = 1
    visualize_single_eeg(data_path, start_time_point, end_time_point,
                        channel)
