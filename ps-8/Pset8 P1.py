import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft

def get_note(file):
    data = np.loadtxt(file)
    t = np.array(range(len(data))) / 44100
    print(data)
    plt.plot(t, data, linewidth=0.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Waveform")
    plt.show()
    return data

piano = get_note("/Users/apple/Desktop/piano.txt")
trumpet = get_note("/Users/apple/Desktop/trumpet.txt")

def fft_note(data):
    c = rfft(data)
    c = np.abs(c[:10000])
    print(c)
    x = np.array(range(10000))
    plt.plot(x, c, linewidth=0.5)
    plt.xlabel("$n$-th coefficient")
    plt.ylabel("Magnitude")
    plt.title("FFT Result")
    plt.show()

fft_note(piano)
fft_note(trumpet)