"""Analysing datafiles from OFDR device.
"""


import os

# os.chdir('C:/Users/Iter/PycharmProjects/loaddata')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')
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

# ##patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs


    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new


# ##patch end###


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


# V_I = loadtxt('EWOFS_fig3_saved.dat',unpack=True, usecols=[0])
mydata = pd.read_table('Data1//compensation_pol1_Upper.txt')
# DataIN = loadtxt('Data1\compensation_pol1_Upper.txt',unpack=True)
time = mydata['Time (ns)']
signal = mydata['Amplitude (dB)']
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(time, signal, lw='1')

mydata2 = pd.read_table('Data1//compensation_pol2_Upper.txt')
time = mydata2['Time (ns)']
signal = mydata2['Amplitude (dB)']
ax.plot(time, signal, lw='1')

mydata3 = pd.read_table('Data1//compensation_pol3_Upper.txt')
time = mydata3['Time (ns)']
signal = mydata3['Amplitude (dB)']
ax.plot(time, signal, lw='1')

# test

plt.show()
