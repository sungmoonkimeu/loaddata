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
from scipy.signal import find_peaks

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

# 1st measurement

a = arange(1, 16, 1)
for nn in a:
    if 7 < nn < 12:
        fn = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + " turn")
        fn = path_dir + "//" + str(nn) + "t_90deg_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + ".25 turn")
        fn = path_dir + "//" + str(nn) + "t_180deg_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + ".5 turn")
        fn = path_dir + "//" + str(nn) + "t_270deg_Upper_edited.txt"
        list_fn = np.append(list_fn, fn)
        legend = np.append(legend, str(nn) + ".75 turn")
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
'''
'''
# 2nd measurement
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
maxdatay1 = np.array([])
maxdatay2 = np.array([])
maxdatax1 = np.array([])
maxdatax2 = np.array([])

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
    #xdata = np.linspace(0, fs, len(fdata))
    xdata = np.fft.fftfreq(len(fdata))*fs

    ax[nn].plot(xdata, 2*abs(fdata), lw='1', label=legend[nn])
    ax[nn].set(xlim=(0, 30), ylim=(0, 5e-13))
    ax[nn].legend(loc="upper right")
    indexes, _ = find_peaks(abs(fdata[0:200]), distance=1, height=0.2e-13)
    print(nn, indexes)

ax[-1].set_xlabel('Frequency (1/m)')
ax[int(len(ax)/2)].set_ylabel('FFT')

yy = np.array([16,15,13,11,9, 7, 6, 4, 4, 4, 3,      5, 5,   5, 6, 8, 9, 11])
yy2 = np.array([18, 14, 11, 8,8, 7,     5, 5,  5, 5,  5, 6, 6, 8, 8, 9,  9, 11, 15, 18])
# for 1st measurement
maxdatax1 = np.array([1, 2, 3, 4, 5, 6, 7, 8,8.25, 8.5,8.75, 11.25,11.5,11.75,12,13,14,15])
#maxdatay1 = 1 / np.array([16,15,13,11,9, 7, 6, 4, 4,  3,   4,   4, 5,   6, 8, 9, 11])
for nn in yy:
    maxdatay1 = np.append(maxdatay1, 1/xdata[nn])
maxdatax2 = np.array(    [5,6,7,8,8.25,8.5,9,9.25, 9.5,9.75,10,10.25,10.5,11,11.25,11.5,11.75,12, 13, 14])
#maxdatay2 = 2 / xdata[7,  7,   5, 5,  5,   5, 6,   8, 9,   11, 15, 18]
for nn in yy2:
    maxdatay2 = np.append(maxdatay2, 2/xdata[nn])

'''
# for 2nd measurement
maxdatax1 = np.array([10.5, 11, 12, 13, 14])
maxdatay1 = 1 / np.array([4, 4, 6, 8, 9])
maxdatax2 = np.array([9, 9.5, 10, 10.5, 11, 11.5])
maxdatay2 = 2 / np.array([5, 5, 5, 6, 8, 9])
'''
ax2.scatter(maxdatax1, maxdatay1)
ax2.scatter(maxdatax2, maxdatay2)

twist = np.arange(1, 16, 0.01)
Lf = 0.65        # fiber length
LB0 = 0.27          # intrinsic beatlength (estimation)
LB1 = 0.32
SP0 = 0.08       # spin  r atio [/m]      (estimation)
STR0 = 2*pi/SP0
STR = 2*pi*twist/Lf
g = 0.15
#Lb_a = 2*pi/sqrt((2*pi/LB0)**2 + (2*STR)**2)
Lb_a = 2*pi/sqrt((2*pi/LB0)**2 + 4*(STR0 - (1-g)*STR)**2)
Lb_b = 2*pi/sqrt((2*pi/LB1)**2 + 4*(STR0 - (1-g)*STR)**2)
ax2.plot(twist, Lb_a)
ax2.plot(twist, Lb_b)

fig2, ax2 = plt.subplots(figsize=(6, 5))

Lf = 0.58        # fiber length
LB0 = 0.27
LB1 = 0.30          # intrinsic beatlength (estimation)
SP0 = 0.072       # spin  r atio [/m]      (estimation)
STR0 = 2*pi/SP0
STR = 2*pi*twist/Lf
g = 0.15
#Lb_a = 2*pi/sqrt((2*pi/LB0)**2 + (2*STR)**2)
Lb_c = 2*pi/sqrt((2*pi/LB0)**2 + 4*(STR0 - (1-g)*STR)**2)
Lb_d = 2*pi/sqrt((2*pi/LB1)**2 + 4*(STR0 - (1-g)*STR)**2)
ax2.scatter(maxdatax1, maxdatay1)
ax2.scatter(maxdatax2, maxdatay2)
#ax2.plot(twist, Lb_c)
ax2.plot(twist, Lb_d)

plt.show()
