"""Analysing datafiles from Polarimeter device
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
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from scipy.interpolate import interp1d
import matplotlib.transforms
import matplotlib.pylab as pl
from matplotlib.colors import rgb2hex

import pandas as pd
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Computer Modern'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['font.sans-serif']='Comic Sans MS'

from tkinter import Tk, filedialog
import os
import sys

print(os.getcwd())
print(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '\My_library')


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


if __name__ == '__main__':

    # figure setting
    plt.close("all")
    plt_fmt, plt_res = '.png', 330  # 330 is max in Word'16
    plt.rcParams["axes.titlepad"] = 5  # offset for the fig title
    # plt.rcParams["figure.autolayout"] = True  # tight_layout
    #  plt.rcParams['figure.constrained_layout.use'] = True  # fit legends in fig window
    fsize = 9
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
    plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('legend', fontsize=fsize - 1)  # f-size legend
    plt.rc('axes', titlesize=11)  # f-size of the axes title (??)
    plt.rc('figure', titlesize=11)  # f-size of the figure title

    # SOP change
    fig, ax = plt.subplots(2,1,figsize=(8.5 / 2.54, 7 / 2.54))
    plt.subplots_adjust(left=0.2, bottom=0.22)


    for mm in range(1):
        # Folder select
        cwd = os.getcwd()
        rootdirectory = os.path.dirname(os.getcwd())
        root = Tk()  # pointing root to Tk() to use it as Tk() in program.
        root.withdraw()  # Hides small tkinter window.
        root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite of selection.
        path_dir = filedialog.askdirectory(initialdir=rootdirectory)  # Returns opened path as str
        file_list = os.listdir(path_dir)

        file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))
        for nn,file_name in enumerate(file_list):
            file_path = path_dir + "/" +file_name
            data = pd.read_table(file_path)
            time = data['Time (ns)']
            #signal = 10 ** (data['Amplitude (dB)'] / 10)
            signal = data['Amplitude (dB/mm)']
            if file_name.split("_")[1] == 'APCAIR':
                ax[0].plot(time / 10, signal, lw='1', label='APC-Air' + str(int(nn/2)+1))
            else:
                if nn > 5:
                    ax[1].plot(time / 10, signal, lw='1', label='APCAPC' + str(nn-5))
            ax[0].legend(loc="upper right")
            ax[1].legend(loc="upper right")
            ax[0].set(xlim=(-1, 10))
            ax[1].set(xlim=(-1, 10))

            ax[1].set_xlabel('Distance (m)')
            ax[0].set_ylabel('Amplitude (dB/mm)')
            ax[1].set_ylabel('Amplitude (dB/mm)')
            ax[0].grid(ls='--', lw=0.5)
            ax[1].grid(ls='--', lw=0.5)

            Ptotal= 10*np.log10((10 ** (signal / 10)).sum())
            l_int = 100e-3 # 10 cm integration length
            length = time/10
            l0 = 4.8
            x_l0 = next(x[0] for x in enumerate(length) if x[1] > l0)
            x_l1 = next(x[0] for x in enumerate(length) if x[1] > l0+l_int)
            x_l2 = next(x[0] for x in enumerate(length) if x[1] > l0+l_int*2)
            lb0 = 8
            x_b0 = next(x[0] for x in enumerate(length) if x[1] > lb0)
            x_b1 = next(x[0] for x in enumerate(length) if x[1] > lb0+l_int)

            P1 = ((10 ** (signal[x_l0:x_l1] / 10)).sum())
            P2 = ((10 ** (signal[x_l1:x_l2] / 10)).sum())
            PB = ((10 ** (signal[x_b0:x_b1] / 10)).sum())

            #RL = 10*np.log10((P2-PB) -(P1-PB)/2)
            RL = 10*np.log10((P2-PB)-(P1-PB)/2)
            print(file_name, ":", RL)


    plt.show()

