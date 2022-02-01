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

    y0 = (max(signal1) - min(signal1))*10   # Acceleration
    y1 = (max(signal0) - min(signal0))      # peak-peak voltage
    y2 = y0 / (frequency ** 2) * 1000       # displacement

    return np.array([y0, y1, y2])


if __name__ == '__main__':

    path = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//'
    folder1 = 'Const_disp_OSC2_edited//'
    folder2 = 'Const_acc_OSC2_edited//'

    folder3 = 'Const_acc_Polarimeter_edited//'
    folder4 = 'Const_disp_Polarimeter2_edited//'
    freq = np.arange(10, 31, 1)

    y_array = np.array([0, 0, 0])

    for nn in freq:
        filename1 = path + folder1 + 'scope_' + str(nn-10) + '_edited.txt'
        #print(filename)
        y_array = read_shakersignal(filename1, nn)
        print(y_array)

        filename2 = path + folder3 + str(nn) + 'Hz_1_edited.txt'
        S = read_SOP(filename2)
        azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha = sqrt(azi**2 + ellip**2) * 180/pi
        print(alpha)

        filename3 = path + folder2 + 'scope_' + str(nn-10) + '_edited.txt'
        #print(filename)
        y_array = read_shakersignal(filename1, nn)
        print(y_array)

        filename4 = path + folder4 + str(nn) + 'Hz_edited.txt'
        S = read_SOP(filename2)
        azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha = sqrt(azi**2 + ellip**2) * 180/pi
        print(alpha)


plt.show()

