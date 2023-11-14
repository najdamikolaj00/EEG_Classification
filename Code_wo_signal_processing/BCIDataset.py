"""
This file is based on the preprocess.py file at https://github.com/Altaheri/EEG-ATCNet,
however this is based on PyTorch not Tensorflow and its architecture is more readable and not dependent on the
training-test set dependent.
"""
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from pre_processing.Preprocessor import Preprocessor, default_preprocessor


class BCIDataset(Dataset):
    def __init__(self, data_paths, preprocessor: Preprocessor = None):
        """
        Custom dataset class for EEG (Electroencephalography) data.

        Parameters:
            data_path (str): Path to the EEG data file.
            is_standard (bool): Whether to standardize the data. Default is True.
        """
        if preprocessor is None:
            preprocessor = default_preprocessor
        self.data_paths = data_paths
        self.X, self.y = [], []
        self.get_data(preprocessor)
        # Subtract 1 from labels for 0-based indexing
        # self.y = (self.y - 1).to(torch.long)

    def get_data(self, preprocessor: Preprocessor):
        """
        Load and preprocess EEG data.

        Parameters:
            path (str): Path to the EEG data file.
            is_standard (bool): Whether to standardize the data. Default is True.

        Returns:
            torch.Tensor: Processed input data.
            torch.Tensor: Processed labels.
        """
        # Constants
        n_channels = 1
        fs = 250  # Sampling rate
        t1 = int(1.5 * fs)  # Start time point
        t2 = int(6 * fs)  # End time point
        T = t2 - t1  # Length of the MI trial (samples or time_points)

        for path in self.data_paths:
            # Load raw data and preprocess
            X, y = self.load_data(path)

            # Select time window and reshape data
            self.N, self.N_ch, _ = X.shape
            X = X[:, :, t1:t2].reshape(self.N, n_channels, self.N_ch, T)

            # Standardize the data if required
            X = preprocessor.preprocess(X)

            y = y - 1
            self.X.append(X)
            self.y.append(y)

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        return self.X, self.y

    @staticmethod
    def load_data(data_path):
        """
        Load EEG data from the given file.

        Parameters:
            data_path (str): Path to the EEG data file.

        Returns:
            np.ndarray: Loaded EEG data.
        """
        # Constants
        n_channels = 22
        n_tests = 6 * 48
        window_length = 7 * 250

        # Initialize arrays for data and labels
        class_return = np.zeros(n_tests)
        data_return = np.zeros((n_tests, n_channels, window_length))

        NO_valid_trial = 0

        # Load data from the provided file
        a = sio.loadmat(data_path)
        a_data = a["data"]

        # Process loaded data
        for ii in range(a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]

            for trial in range(0, a_trial.size):
                if a_artifacts[trial]:
                    continue
                data_return[NO_valid_trial, :, :] = np.transpose(
                    a_X[
                        int(a_trial[trial]) : (int(a_trial[trial]) + window_length),
                        :n_channels,
                    ]
                )
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial += 1

        return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]

    def __len__(self):
        """Get the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Get a sample (input, label) from the dataset."""
        return self.X[idx], self.y[idx]
