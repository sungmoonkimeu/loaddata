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

# os.chdir('C:/Users/Iter/PycharmProjects/loaddata')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')

path_dir = 'Data3_edited'
file_list = os.listdir(path_dir)

'''
for nn in range(len(file_list)):
    fn = path_dir + "//" + file_list[nn]
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(time, signal, lw='1', label=file_list[nn])
    ax.legend(loc="upper right")
    ax.set(xlim=(124, 134), ylim=(-135, -120))
plt.show()
'''

a = arange(30, 360, 30)
count = 0
fig, ax = plt.subplots(len(a), figsize=(6, 5))

for nn in a:

    fn2 = path_dir + "//" + "9turns_" + str(nn) + "deg_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    ax[count].plot(time, signal, lw='1', label="9turns_"+str(nn)+"deg")
    ax[count].legend(loc="upper right")
    ax[count].set(xlim=(124, 134), ylim=(-135, -120))
    count = count+1

plt.show()
