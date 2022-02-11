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

#foldername = 'Const_appl_vol_Polarimeter'
#foldername = '0_RHC_losen'
#foldername = '2_RHC'
#foldername = 'Const_disp_Polarimeter2'
V_foldername = ['2_LP0_loosen', '1_LP0']
#V_foldername = ['1_LP0', '1_LP45', '1_RHC_fasten']
#V_foldername = ['Const_volt_LP90_Polarimeter', 'Const_volt_LP45_Polarimeter', 'Const_volt_RHC_Polarimeter']
V_label = ['Loosen', 'Tighten']
#V_marker = ['^', 'o', 'x']

issaved = False
for n_iter, foldername in enumerate(V_foldername):
    #path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
    path_dir = os.getcwd() + '//Data_Vib_3_(Hibi_loosen_fasten)//' + foldername + '_edited'

    #path_dir = os.getcwd() + '//Data_Vib_2_(Hibi_losen_fasten)//' + foldername + '_edited'
    #path_dir = os.getcwd() + '//Data_Vib_3_(Hibi_loosen_fasten)//' + foldername + '_edited'
    file_list = os.listdir(path_dir)


    Ev = Jones_vector('Output_J')
    Sv = create_Stokes('Output_S')
    Out = create_Stokes('Output_S2')

    #fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
    frequency = arange(10, 31, 1)
    #frequency = np.array([30,31])

    diff_azi_V = np.ones(len(file_list))
    diff_ellip_V = np.ones(len(file_list))
    fig, ax = plt.subplots(4, figsize=(12/2.54, 9/2.54))
    fig.set_dpi(91.79)  # DPI of My office monitor

    plt.subplots_adjust(left=0.25, bottom=0.17, right=0.93, top=0.93, wspace=0.2, hspace=0.233)

    max_diff_S = np.zeros([2,3])
    mean_S = np.zeros([2,3])
    for nn in range(len(file_list)):
        fn2 = path_dir + "//" + file_list[nn]
        count = 0
        cstm_color = ['c', 'm', 'y', 'k', 'r']

        #    fn2 = path_dir + "//10Hz_edited.txt"
        data = pd.read_table(fn2, delimiter=r"\s+")
        time = pd.to_numeric(data['Index']) / 10000
        S0 = pd.to_numeric(data['S0(mW)'])
        S1 = pd.to_numeric(data['S1'])
        S2 = pd.to_numeric(data['S2'])
        S3 = pd.to_numeric(data['S3'])

        Sn = np.ones((len(S0)))
        SS = np.vstack((Sn, S1, S2, S3))
        Out = Sv.from_matrix(SS.T)

        #draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])

        azi_V = Out.parameters.azimuth()
        ellip_V = Out.parameters.ellipticity_angle()
        diff_azi_V[nn] = azi_V.max() - azi_V.min()
        diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()

        if nn == len(file_list)-2:
            ax[0].plot(time[::2], S0[::2], 'k')
            ax[1].plot(time[::2], S1[::2], 'k')
            ax[2].plot(time[::2], S2[::2], 'k')
            ax[3].plot(time[::2], S3[::2], 'k')
            print(S1.max(), S1.min())
            print(S2.max(), S2.min())
            print(S3.max(), S3.min())
            ax[0].set(xlim=(0, 0.3), ylim=(1.52, 1.55))
            ax[0].set(xticklabels=[])
            ax[1].set(xticklabels=[])
            ax[2].set(xticklabels=[])
            if issaved is False:
                ax[1].set(xlim=(0, 0.3), ylim=(-0.96446, -0.95445))
                ax[2].set(xlim=(0, 0.3), ylim=(-0.088255, -0.070255))
                ax[3].set(xlim=(0, 0.3), ylim=(0.25029, 0.29029))
            else:
                ax[1].set(xlim=(0, 0.3), ylim=(-0.84853, -0.83853))
                ax[2].set(xlim=(0, 0.3), ylim=(0.044285, 0.062285))
                ax[3].set(xlim=(0, 0.3), ylim=(0.514515, 0.554515))
            issaved = True

    for nn in range(len(ax)):
        ax[nn].set_ylabel("S"+str(nn))
        #if nn > 0:

    ax[3].set_xlabel("Time (s)")
    fig.align_ylabels()

    #plt.rc('text', usetex=True)
    #r'$\phi$'


plt.show()


