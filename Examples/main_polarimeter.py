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


foldername = 'Data_pol_laser'

path_dir = os.getcwd() + '//Data_Vib_0_(Polarimeter)//' + foldername + '_edited'
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
fig, ax = plt.subplots(4, figsize=(6, 5))
Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')

for nn in range(len(file_list)):

    fn2 = path_dir + "//" + file_list[nn]
    count = 0
    cstm_color = ['c', 'm', 'y', 'k', 'r', 'g', 'b']

#    fn2 = path_dir + "//10Hz_edited.txt"
    data = pd.read_table(fn2, delimiter=r"\s+")
    time = pd.to_numeric(data['Index'])/10000
    S0 = pd.to_numeric(data['S0(mW)'])
    S1 = pd.to_numeric(data['S1'])
    S2 = pd.to_numeric(data['S2'])
    S3 = pd.to_numeric(data['S3'])

    Sn = np.ones((len(S0)))
    #SS = np.vstack((S0, S1, S2, S3))
    SS = np.vstack((Sn, S1, S2, S3))

    Out = Sv.from_matrix(SS.T)
    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 5])

    ax[0].plot(time, S0)
    #ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[1].plot(time, S1)
    #ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[2].plot(time, S2)
    #ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[3].plot(time, S3)
    #ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))

    '''
    for nn in a:
    
        fn2 = path_dir + "//" + "9turns_" + str(nn) + "deg_Upper_edited.txt"
        data = pd.read_table(fn2)
        time = data['Time (ns)']
        signal = data['Amplitude (dB)']
        ax[count].plot(time, signal, lw='1', label="9turns_"+str(nn)+"deg")
        ax[count].legend(loc="upper right")
        ax[count].set(xlim=(124, 134), ylim=(-135, -120))
        count = count+1
    '''

plt.show()
