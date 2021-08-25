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

#foldername = 'Data_1308//spunfiber//1st'
foldername = 'Data_1308//withouthspun'

path_dir = os.getcwd() + '//Data_Twsiting_(OFDR)//' + foldername + '_edited'
file_list = os.listdir(path_dir)


def plotsignal(length, signal, ax, xmin, xmax, ymin, ymax, legend=None):

    ax.plot(length, signal, lw='1', label=legend)
    ax.legend(loc="upper left")
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


## Create a list of files of interest

list_fn = []
legend = []
b = arange(2, 0, -1)
for nn in b:
    fn = path_dir + "//" + str(nn) + "t_c_Upper_edited.txt"
    list_fn = np.append(list_fn, fn)
    legend = np.append(legend, str(-nn)+" turn")
fn = path_dir + "//lin_biref_base_mes_Upper_edited.txt"
list_fn = np.append(list_fn, fn)
legend = np.append(legend, "0 turn")
a = arange(1, 8, 1)
for nn in a:
    # fn2 = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
    fn = path_dir + "//" + str(nn) + "t_ac_Upper_edited.txt"
    list_fn = np.append(list_fn, fn)
    legend = np.append(legend, str(nn)+" turn")

fig, ax = plt.subplots(len(list_fn), figsize=(6, 5))
for nn, fn in enumerate(list_fn):
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10

    xmin = 11.04 # 10.9
    xmax = 11.6  # 12.0
    ymin = -140
    ymax = -125
    '''
    xmin = 3.5
    xmax = 4.6
    ymin = -134
    ymax = -116
    '''
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

for nn, fn in enumerate(list_fn):
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10

    xmin = 10.9
    xmax = 12.0
    ymin = -140
    ymax = -125

    xi = 11.04
    xf = 11.6

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
    ax[nn].set(xlim=(0, 20), ylim=(0, 2e-14))
    ax[nn].legend(loc="upper right")

    ax2.plot(xdata, 2*abs(fdata), label=legend[nn])
    ax2.legend(loc="upper right")
    ax2.set(xlim=(0, 20), ylim=(0, 2e-14))

plt.show()
