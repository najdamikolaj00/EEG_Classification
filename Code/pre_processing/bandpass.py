import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftfreq
from scipy.signal import butter, lfilter
from sympy import fft


def _butter_bandpass(low, high, fs, order=5):
    b, a = butter(order, (low, high), btype="band", fs=fs)
    return b, a


def bandpass(data, low, high, fs, order=5):
    b, a = _butter_bandpass(low, high, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    freq = 2
    sample_rate = 100  # Hertz
    duration = 1  # Seconds
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    filtered = bandpass(y, 15, 25, fs=sample_rate)
    x = fftfreq(duration * sample_rate, 1 / sample_rate)
    y = np.abs(fft(filtered))
    plt.plot(x, y)
    plt.show()
