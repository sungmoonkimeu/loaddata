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


from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse


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


def read_SOP(filepath):
    # reading every file in a folder
    # calculating maximum SOP change for each file
    # output: SOP change for given frequency range

    data = pd.read_table(filepath, delimiter=r"\s+")
    time = pd.to_numeric(data['Index']) / 10000
    S0 = pd.to_numeric(data['S0(mW)'])
    S1 = pd.to_numeric(data['S1'])
    S2 = pd.to_numeric(data['S2'])
    S3 = pd.to_numeric(data['S3'])

    Sv = create_Stokes('Output_S')
    SS = np.vstack((S0, S1, S2, S3))

    return Sv.from_matrix(SS.T)

def read_shakersignal(filepath, frequency):

    data = pd.read_table(filepath, sep=",")
    time = data['second']
    signal0 = data['Volt']
    signal1 = data['Volt.1']

    y0 = (max(signal0) - min(signal0))      # peak-peak voltage
    y1 = (max(signal1) - min(signal1)) * 10  # Acceleration
    y2 = y0 / (frequency ** 2) * 1000       # displacement

    return np.array([y0, y1, y2])


if __name__ == '__main__':

    path = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//'
    folder1 = 'Const_acc_OSC2_edited//'
    folder2 = 'Const_acc_Polarimeter_edited//'

    folder3 = 'Const_disp_OSC2_edited//'
    folder4 = 'Const_disp_Polarimeter2_edited//'



    freq = np.arange(10, 31, 1)

    displacement = np.zeros(len(freq))
    acceleration = np.zeros(len(freq))
    alpha1 = np.zeros(len(freq))
    alpha2 = np.zeros(len(freq))

    for nn in freq:
        # Const displacement
        filename1 = path + folder1 + 'scope_' + str(nn-10) + '_edited.txt'

        y_array = read_shakersignal(filename1, nn)
        displacement[nn-10] = y_array[2]

        filename2 = path + folder2 + str(nn) + 'Hz_1_edited.txt'
        S = read_SOP(filename2)
        delta_azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        delta_ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha1[nn-10] = sqrt(delta_azi**2 + delta_ellip**2) * 180/pi

        # Const acceleration
        filename3 = path + folder3 + 'scope_' + str(nn-10) + '_edited.txt'
        #print(filename)
        y_array = read_shakersignal(filename3, nn)
        acceleration[nn - 10] = y_array[1]

        filename4 = path + folder4 + str(nn) + 'Hz_edited.txt'
        S = read_SOP(filename4)
        delta_azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        delta_ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha2[nn-10] = sqrt(delta_azi**2 + delta_ellip**2) * 180/pi

    fig, ax = plt.subplots(1, 2, figsize=(23/2.54, 8/2.54))
    fig.set_dpi(91.79)  # DPI of My office monitor
    plt.subplots_adjust(left=0.088, bottom=0.155, right=0.91, top=0.93, wspace=0.593, hspace=0.502)
    plt.subplots_adjust(bottom=0.155)

    ms = 4
    ax[0].plot(freq, alpha1, lw='1', label="Max. SOP change", marker='o', color='k', markersize=ms)
    ax[0].set_xlabel('Frequency (Hz)')
    #ax[0].set_ylabel('Max. SOP change (deg)')
    ax[0].set_ylabel(r'$\alpha$ (deg)')
    #ax[0].set_title('Input signal')
    ax[0].set(xlim=(10, 30), ylim=(0, 2))
    lns01 = ax[0].lines
    ax0 = ax[0].twinx()
    lns02 = ax0.plot(freq, displacement, lw='1', label="Displacement", marker='x', color='r', markersize=ms)
    ax0.set_ylabel('Displacement (mm)')
    ax0.set(ylim=(0, 180))

    # added these three lines
    lns = lns01 + lns02
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc=0)

    ax[1].plot(freq, alpha2, lw='1', label="Max. SOP change", marker='o', color='k', markersize=ms)
    ax[1].set_xlabel('Frequency (Hz)')

    #ax[1].set_ylabel('Max. SOP change (deg)')
    ax[1].set_ylabel(r'$\alpha$ (deg)')
    #ax[1].set_title('Measured signal')
    ax[1].set(xlim=(10, 30), ylim=(0, 2))
    lns11 = ax[1].lines
    ax1 = ax[1].twinx()
    lns12 = ax1.plot(freq, acceleration, lw='1', label="Acceleration", marker='^', color='b', markersize=ms)
    ax1.set_ylabel('Acceleration (g)')
    ax1.set(ylim=(0, 40))

    # added these three lines
    lns = lns11 + lns12
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=0)

    fig.align_ylabels()
    #fig.savefig('Constant_Acceleration.png')
    #fig.savefig('Constant_Displacement.png')

plt.show()

