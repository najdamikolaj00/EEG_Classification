from typing import NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftfreq
from scipy.signal import butter, lfilter
from sympy import fft


class BandpassArgs(NamedTuple):
    low: int
    high: int
    fs: int


def _butter_bandpass(bandpass_args: BandpassArgs, order=5):
    b, a = butter(
        order,
        (bandpass_args.low, bandpass_args.high),
        btype="band",
        fs=bandpass_args.fs,
    )
    return b, a


def bandpass(data, bandpass_args: BandpassArgs, order=5):
    b, a = _butter_bandpass(bandpass_args, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    freq = 2
    sample_rate = 100  # Hertz
    duration = 1  # Seconds
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    filtered = bandpass(y, BandpassArgs(15, 25, sample_rate))
    x = fftfreq(duration * sample_rate, 1 / sample_rate)
    y = np.abs(fft(filtered))
    plt.plot(x, y)
    plt.show()
