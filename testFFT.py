import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

fs = 10000
t = np.arange(0, 100, 1/fs)
sig = np.sin(2*np.pi * 100 * t)
sp = np.fft.fft(sig)
trange = np.linspace(0, fs, len(t))

fig, ax = plt.subplots(3, figsize=(6, 5))
ax[0].plot(trange, np.abs(sp)/len(sig)*2, label="FFT")
ax[1].plot(trange, ((np.abs(sp)/len(sig)*2)**2)/(fs/len(sp)), label="PSD")

f, Pxx_den = welch(sig, fs)
ax[2].plot(f, Pxx_den, label='using Welch func')

plt.show()
