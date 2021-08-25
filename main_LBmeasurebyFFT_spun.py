"""Analysing datafiles from OFDR device.
"""

import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack, savetxt, loadtxt
from numpy.linalg import norm, eig, matrix_power
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from scipy.interpolate import interp1d
from scipy.signal import welch

import matplotlib.transforms
import pandas as pd
import os

# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Computer Modern'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['font.sans-serif']='Comic Sans MS'

# noinspection PyPep8Naming
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def switch_osfolder(path1, path2):
    try:
        if os.path.exists(path1):
            os.chdir(path1)
        else:
            os.chdir(path2)
    except OSError:
        print('Error: Changing OS directory')

# Switching OS folder
path2 = 'C:/Users/Iter/PycharmProjects/loaddata'
path1 = 'C:/Users/SMK/PycharmProjects/loaddata'
switch_osfolder(path1, path2)

foldername = 'Data_1308//spunfiber//1st'
#foldername = 'Data_1308//withouthspun'
#foldername = 'Data_1308//spunfiber//2nd'

path_dir = os.getcwd() + '//Data_Twsiting_(OFDR)//' + foldername + '_edited'
file_list = os.listdir(path_dir)


def plotsignal(length, signal, ax, xmin, xmax, ymin, ymax, legend=None):

    ax.plot(length, signal, lw='1', label=legend)
    ax.legend(loc="upper left")
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


## Create a list of files of interest

list_fn = []
legend = []
'''
a = arange(9, 16, 1)
for nn in a:
    if 8 < nn < 12:
        fn = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + " turn")
        fn = path_dir + "//" + str(nn) + "t_180deg_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + ".5 turn")
    else:
        fn = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + " turn")

'''
a = arange(6, 12, 1)
for nn in a:
    if 7 < nn < 12:
        fn = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + " turn")
        fn = path_dir + "//" + str(nn) + "t_180deg_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + ".5 turn")
    else:
        fn = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + " turn")

fig, ax = plt.subplots(len(list_fn), figsize=(6, 5))
for nn, fn in enumerate(list_fn):
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10

    xmin = 3.5
    xmax = 4.6
    ymin = -134
    ymax = -116

    plotsignal(length, signal, ax[nn], xmin, xmax, ymin, ymax, legend[nn])

ax[-1].set_xlabel('Length (m)')
ax[int(len(ax)/2)].set_ylabel('Power (dB/mm)')
#ax2.set_xlabel('Length (m)')
#ax2.set_ylabel('Power (dB/mm)')


# FFT    ddddddddddddddddddddddddd
# ddddddddddddddddddddddddddddddddd
# fn2 = path_dir + "//lin_biref_base_mes_Upper_edited.txt"
# fn2 = path_dir + "//2t_ac_Upper_edited.txt"

fig, ax = plt.subplots(len(list_fn), figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))
fig3, ax3 = plt.subplots(len(list_fn), figsize=(6, 5))

maxdatay = np.array([])
maxdatax = np.array([])
for nn, fn in enumerate(list_fn):
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10

    xmin = 3.8
    xmax = 4.5
    ymin = -134
    ymax = -116

    xi = 3.8
    xf = 4.5

    data2 = np.array([])
    for mm, x in enumerate(length):
        if xi < x < xf:
            data2 = np.append(data2, 10 ** (signal[mm] / 10))
            #data2 = np.append(data2, signal[mm])
    data2 = data2 - data2.mean()
    fs = 1 / (length[2]-length[1])

    fdata = np.fft.fft(data2)/len(data2)
    xdata = np.linspace(0, fs, len(fdata))
    ax[nn].plot(xdata, 2*abs(fdata), lw='1', label=legend[nn])
    ax[nn].set(xlim=(0, 50), ylim=(0, 5e-13))


    f, pxx_den = welch(data2, fs)
    ax3[nn].plot(f, pxx_den, lw='1', label=legend[nn])
    #ax3[nn].psd(data2, int(len(data2)/1), fs, label=legend[nn])
    ax3[nn].legend(loc="upper right")
    #ax3[nn].set(xlim=(0, 50), ylim=(0, 4e-26))

    maxdatax = np.append(maxdatax, nn)
    maxdatay = np.append(maxdatay, 2/xdata[np.argmax(abs(fdata[0:100]))])
    print(np.argmax(abs(fdata)))

ax2.plot(maxdatax, maxdatay)

plt.show()
