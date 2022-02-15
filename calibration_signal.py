import numpy as np
from numpy import pi
from scipy.signal import butter, filtfilt
from scipy import signal
import matplotlib.pyplot as plt

# Filter requirements.
T = 24             # Sample Period (sec)
fs = 10000.0         # sample rate, Hz
tc = 10e-3          # time constant RC = 1/w0
w0 = 1/tc           # cut off frequency (rad)
cutoff = w0/(2*pi)      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 1       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

t = np.arange(0, T, 1/fs)
# sin wave
sig = np.sin(10*w0*t)
# square wave
sig = -0.5*signal.square(2*pi*1/12*t)+0.5
# Lets add some noise
#noise = 1.5*np.cos(2*20*np.pi*t) + 0.5*np.sin(15*2*np.pi*t)
noise = 0
data = sig + noise


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(t,data)
ax.plot(t, y)

plt.show()
