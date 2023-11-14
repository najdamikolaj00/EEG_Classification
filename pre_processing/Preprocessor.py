import numpy as np
from sklearn.preprocessing import StandardScaler

from pre_processing.bandpass import BandpassArgs, bandpass
from pre_processing.classes.ClassTrails import ClassTrails
from pre_processing.csp import gen_csp
from pre_processing.isoelectric_line_removal import isoelectric_line_removal
from pre_processing.sax import SaxArgs, get_sax


class Preprocessor:
    n_channels = 22
    n_dims = 1
    n_samples = 1125
    n_trains = 10
    mock_signal = np.random.rand(n_trains, n_dims, n_channels, n_samples)
    mock_targets = np.array(tuple(map(int(4).__rmod__, range(n_trains))))

    def __init__(
        self,
        standardize=True,
        isoelectric_line_removal=True,
        sax: SaxArgs = None,
        bandpass_parameters: BandpassArgs = None,
        csp=True,
    ):
        self.csp = csp
        self.isoelectric_line_removal = isoelectric_line_removal
        self.bandpass_parameters = bandpass_parameters
        self.sax = sax
        self.standardize = standardize
        self.csp_applier = None

    def get_dims(self):
        return self.preprocess(self.mock_signal, self.mock_targets).shape[1:]

    def preprocess(self, X, y):
        if self.isoelectric_line_removal or self.bandpass_parameters:
            X = isoelectric_line_removal(X)
        if self.bandpass_parameters:
            X = bandpass(X, self.bandpass_parameters)
        csp = None
        if self.csp:
            class_trails = tuple(
                ClassTrails(
                    class_, X[np.where(self.mock_targets == class_)][:, -1, :, :]
                )
                for class_ in range(4)
            )
            self.csp_applier = gen_csp(class_trails)
            csp = np.array(tuple(map(self.csp_applier.apply, X)))[
                :, np.newaxis, np.newaxis, :
            ]
        if self.standardize:
            X = self.standardize_data(X)
        if self.sax:
            saxs = get_sax(X, self.sax.segment_length)
            if self.sax.sax_only:
                X = saxs
            else:
                X = np.append(X, saxs, axis=-1)
        # if csp is not None:
        #     X = np.append(X, csp, axis=-1)
        return X

    def standardize_data(self, X):
        """
        Standardize the data using StandardScaler.

        Returns:
            np.ndarray: Standardized data.
        """
        # X :[Trials, Filters=1, Channels, Time points]
        for j in range(self.n_channels):
            scaler = StandardScaler()
            scaler.fit(X[:, 0, j, :])
            X[:, 0, j, :] = scaler.transform(X[:, 0, j, :])

        return X


default_preprocessor = Preprocessor(
    sax=SaxArgs(100, False), bandpass_parameters=BandpassArgs(5, 15, 250)
)
if __name__ == "__main__":
    default_preprocessor.get_dims()
