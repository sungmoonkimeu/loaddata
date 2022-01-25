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

plt.rcParams['font.size'] = 12


def read_shakerinputsignal(foldername, fig=None):
    path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
    file_list = os.listdir(path_dir)

    count = 0
    x0 = zeros(len(file_list))
    y0 = zeros(len(file_list))
    y1 = zeros(len(file_list))
    for nn in range(len(file_list)):

        fn2 = path_dir + "//" + "scope_" + str(nn) + "_edited.txt"
        data = pd.read_table(fn2, sep=",")
        time = data['second']
        signal0 = data['Volt']
        signal1 = data['Volt.1']

        x0[nn] = 10 + nn
        y0[nn] = (max(signal1) - min(signal1))*10
        y1[nn] = (max(signal0) - min(signal0))
        print(y1[nn])

    fig, ax = plt.subplots(2, figsize=(4, 6))

    fig.set_dpi(91.79)  # DPI of My office monitor
    plt.subplots_adjust(left=0.19, bottom=0.13, right=0.833, top=0.93, wspace=0.2, hspace=0.52)
    plt.subplots_adjust(bottom=0.155)

    ms = 4
    ax[0].plot(x0, y1, lw='1', label="ff", marker='o', color='k', markersize=ms)
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Peak-peak voltage (V)')
    ax[0].set_title('Input signal')
    ax[0].set(xlim=(10, 30), ylim=(0, 12))

    ax[1].plot(x0, y0, lw='1', label="Acceleration", marker='o', color='k', markersize=ms)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Acceleration (g)')
    ax[1].set_title('Measured signal')
    ax[1].set(xlim=(10, 30), ylim=(0, 30))
    lns1 = ax[1].lines
    ax2 = ax[1].twinx()
    lns2 = ax2.plot(x0, y0 / (x0 ** 2) * 1000, lw='1', label="Displacment", marker='o', color='r', markersize=ms)
    ax2.set_ylabel('Displacement (mm)')
    ax2.set(ylim=(0, 100))

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=0)

    fig.align_ylabels()
    #fig.savefig('Constant_Acceleration.png')
    #fig.savefig('Constant_Displacement.png')
    return fig


if __name__ == '__main__':

    foldername = 'Const_disp_OSC2'

    fig = read_shakerinputsignal(foldername)
    fig.savefig(foldername)

    foldername = 'Const_acc_OSC2'

    fig = read_shakerinputsignal(foldername)
    fig.savefig(foldername)

plt.show()
