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


import os

os.chdir('C:/Users/Iter/PycharmProjects/loaddata')
#os.chdir('C:/Users/SMK/PycharmProjects/loaddata/')


path_dir = 'Data_Vib/Acceleration2_edited'
file_list = os.listdir(path_dir)

#a = arange(30, 361, 30)
count = 0
#fig, ax = plt.subplots(len(a), figsize=(6, 5), left=0.093, bottom = 0.07, right = 0.96, top = 0.967, wspace = 0.2, hspace = 0 )
fig, ax = plt.subplots(len(file_list), figsize=(6, 5))
plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)
x0 = zeros(len(file_list))
y0 = zeros(len(file_list))

for nn in range(len(file_list)):

    fn2 = path_dir + "//" + "saf" + str(nn+1) + "_edited.txt"
    data = pd.read_table(fn2, sep=",")
    time = data['second']
    signal0 = data['Volt']
    signal1 = data['Volt.1']
    ax[nn].plot(time, signal0, lw='1', label="applied voltage")
    ax[nn].plot(time, signal1*10, lw='1', label="Accelerometer signal")
    ax[nn].legend(loc="upper left")
    #ax[count].set(xlim=(11.8, 13.7), ylim=(-135, -120))
    #ax[count].set(xlim=(11.8, 13.7), ylim=(10**(-13.7), 10**(-12.5)))
    x0[nn] = 10 + nn*2
    y0[nn] = (max(signal1) - min(signal1))*10

fig, ax = plt.subplots(figsize=(6, 3))
#plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)
plt.subplots_adjust(bottom=0.155)
#ax.set(xlim=(11.8, 13.7), ylim=(10 ** (-13.7), 10 ** (-12.5)))
#ax.legend(loc="upper left")
#plt.rc('text', usetex=True)
ax.plot(x0, y0, lw='1', label="ff")
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Acceleration (g)')

fig2, ax2 = plt.subplots(figsize=(6, 3))
plt.subplots_adjust(bottom=0.155)
ax2.plot(x0, y0/(x0**2)*1000, lw='1', label="ff")
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Displacement (mm)')

plt.show()
