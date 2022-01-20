"""Analysing datafiles from Oscilloscope device.
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

## CHOOSE A FOLDER THAT CONTAINS DATA
## CHOOSE A FOLDER THAT CONTAINS DATA
## CHOOSE A FOLDER THAT CONTAINS DATA

foldername = 'Const_acc_OSC2'
#foldername = 'Const_disp_OSC2'
#foldername = '010921 Vibration in second part_OSC//3.5V input'

## CHOOSE A FOLDER THAT CONTAINS DATA
## CHOOSE A FOLDER THAT CONTAINS DATA
## CHOOSE A FOLDER THAT CONTAINS DATA

# Switching OS folder
path2 = 'C:/Users/Iter/PycharmProjects/loaddata'
path1 = 'C:/Users/SMK/PycharmProjects/loaddata'
import os
def switch_osfolder():
    try:
        if os.path.exists(path1):
            os.chdir(path1)
        else:
            os.chdir(path2)
    except OSError:
        print('Error: Changing OS directory')

switch_osfolder()

path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
file_list = os.listdir(path_dir)

#a = arange(30, 361, 30)
count = 0
#fig, ax = plt.subplots(len(a), figsize=(6, 5), left=0.093, bottom = 0.07, right = 0.96, top = 0.967, wspace = 0.2, hspace = 0 )
fig, ax = plt.subplots(len(file_list), figsize=(6, 5))
plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)
x0 = zeros(len(file_list))
y0 = zeros(len(file_list))
y1 = zeros(len(file_list))
for nn in range(len(file_list)):

    fn2 = path_dir + "//" + "scope_" + str(nn) + "_edited.txt"
    #fn2 = path_dir + "//" + "t" + str(nn) + "_edited.txt"
    data = pd.read_table(fn2, sep=",")
    time = data['second']
    signal0 = data['Volt']
    signal1 = data['Volt.1']
    ax[nn].plot(time, signal0, lw='1', label="applied voltage")
    ax[nn].plot(time, signal1*10, lw='1', label="Accelerometer signal")
    ax[nn].legend(loc="upper left")
    #ax[count].set(xlim=(11.8, 13.7), ylim=(-135, -120))
    #ax[count].set(xlim=(11.8, 13.7), ylim=(10**(-13.7), 10**(-12.5)))
    x0[nn] = 10 + nn
    y0[nn] = (max(signal1) - min(signal1))*10
    y1[nn] = (max(signal0) - min(signal0))
    print(y1[nn])

fig, ax = plt.subplots(3, figsize=(4, 6))
plt.subplots_adjust(left=0.138, bottom=0.13, right=0.926, top=0.945, wspace=0.2, hspace=0.19)
plt.subplots_adjust(bottom=0.155)
#ax.set(xlim=(11.8, 13.7), ylim=(10 ** (-13.7), 10 ** (-12.5)))
#ax.legend(loc="upper left")
#plt.rc('text', usetex=True)
ax[0].plot(x0, y0, lw='1', label="ff", marker='o')
ax[0].set_xlabel('Freq. (Hz)')
ax[0].set_ylabel('Acc. (g)')
#ax[0].set(xlim=(10, 30), ylim=(0, 10))
ax[0].set(xlim=(10, 30), ylim=(0, 20))

#fig2, ax2 = plt.subplots(figsize=(6, 3))
#plt.subplots_adjust(bottom=0.155)
ax[1].plot(x0, y0/(x0**2)*1000, lw='1', label="ff", marker='o', color='r')
ax[1].set_xlabel('Freq. (Hz)')
ax[1].set_ylabel('Disp. (mm)')
ax[1].set(xlim=(10, 30), ylim=(0, 80))

ax[2].plot(x0, y1, lw='1', label="ff", marker='o')
ax[2].set_xlabel('Freq. (Hz)')
ax[2].set_ylabel('Input (Vp-p)')
ax[2].set(xlim=(10, 30), ylim=(0, 10))
fig.align_ylabels()

fig, ax = plt.subplots(2, figsize=(6, 5))
plt.subplots_adjust(left=0.145, bottom=0.117, right=0.76, top=0.967, wspace=0.2, hspace=0.288)
x0 = zeros(len(file_list))
y0 = zeros(len(file_list))
for nn in range(len(file_list)):

    fn2 = path_dir + "//" + "scope_" + str(nn) + "_edited.txt"
    #fn2 = path_dir + "//" + "a" + str(nn) + "_edited.txt"
    data = pd.read_table(fn2, sep=",")
    time = data['second']
    signal0 = data['Volt']
    signal1 = data['Volt.1']
    x0[nn] = 10+nn
    y0[nn] = (max(signal1) - min(signal1))*10

    ax[0].plot(time, signal1*10, lw='1', label=str(x0[nn]) + "Hz")
    #ax[0].legend(loc="upper right")
    #ax[1].legend(loc="upper right")
    ax[0].set(xlim=(0, 0.1))
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Acceleration (g)')

    ax[1].plot(time, signal0, lw='1', label=str(x0[nn]) + "Hz")
    ax[1].set(xlim=(0, 0.1))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Applied voltage (V)')

    #ax[count].set(xlim=(11.8, 13.7), ylim=(10**(-13.7), 10**(-12.5)))
plt.legend(loc="center left", bbox_to_anchor=(1.04, 1.2))

plt.show()
