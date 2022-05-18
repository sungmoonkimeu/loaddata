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

from matplotlib.lines import Line2D

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import basis_calibration


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

#foldername = '//Laser stability test_short_term'
#foldername = '//Laser stability test_2nd'
#foldername = '//Laser stability test_longterm'
#foldername = '//Laser_stability_test_pol_manualPC'
#foldername = '//Laser_stability_test_pol2_manualPC'
#foldername = '//Stability_ManualPC'
#foldername = '//Laser_stability_test_cascadedpol'

#foldername = '//Data_Stability/EDFA_TEST_2004'
foldername = '//Data_Stability/Stability_again2_edited/dd'
#foldername = '//Stability_total_manualPC'

path_dir = os.getcwd() + foldername + '_edited'

file_list = os.listdir(path_dir)

plt.subplots_adjust(left=0.14, bottom=0.112, right=0.93, top=0.93, wspace=0.2, hspace=0)

Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
frequency = arange(10, 15,1)
#frequency = np.array([30, 31])

diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))

max_diff_S = np.zeros([2, 4])
mean_S = np.zeros([2, 4])


fig_1, ax1 = plt.subplots(4, figsize=(6, 5))
fig_2, ax2 = plt.subplots(4, figsize=(6, 5))
tmpax = 0
for nn in range(len(file_list)):

    fn2 = path_dir + "//" + file_list[nn]
    print("filename = ", file_list[nn])
    count = 0
    cstm_color = ['y', 'b', 'r', 'k', 'g']

    #    fn2 = path_dir + "//10Hz_edited.txt"
    data = pd.read_table(fn2, delimiter=r"\s+")
    #time = pd.to_numeric(data['Time(ms)']) / 10000
    #time = np.arange(0, 3600, 1)/3600

    S00 = pd.to_numeric(data['S0(mW)'])
    S1 = pd.to_numeric(data['S1'])
    S2 = pd.to_numeric(data['S2'])
    S3 = pd.to_numeric(data['S3'])

    time = np.arange(0, len(S00), 1) / 1800


    Sn = np.ones((len(S00)))
    SS = np.vstack((Sn[0::2], S1[0::2], S2[0::2], S3[0::2]))
    # SS = np.vstack((Sn[1800::2], S1[1800::2], S2[1800::2], S3[1800::2]))
    #SS = np.vstack((Sn[720::2], S1[720::2], S2[720::2], S3[720::2]))
    #SS = np.vstack((Sn[720:720+ndata:2], S1[720:720+ndata:2], S2[720:720+ndata:2], S3[720:720+ndata:2]))
    Out = Sv.from_matrix(SS.T)

    #draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])
    Out = basis_calibration.calib_basis2(Out)

    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 5])


    S0 = Out.parameters.matrix()[0]
    S1 = Out.parameters.matrix()[1]
    S2 = Out.parameters.matrix()[2]
    S3 = Out.parameters.matrix()[3]



    azi_V = Out.parameters.azimuth()
    ellip_V = Out.parameters.ellipticity_angle()

    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()
    print('maximum SOP change=', sqrt(diff_azi_V[nn]**2 + diff_ellip_V[nn]**2)*180/pi, "deg")


    tmpax = ax1
    tmpfig = fig_1
    if nn == 0:
        #     tmpax = ax1
        #     tmpfig = fig_1
        strlabel = '2nd'
    else:
    #     tmpax = ax2
    #     tmpfig = fig_2
        strlabel= '1st'

    time = time[::2]*20
    S00 = S00[::2]
    tmpax[0].plot(time, S00, label=strlabel)
    tmpax[0].set_ylabel("S" + str(0))
    tmpax[0].legend(loc='upper right')
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[1].plot(time, S1)
    tmpax[1].set_ylabel("S" + str(1))
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[2].plot(time, S2)
    tmpax[2].set_ylabel("S" + str(2))
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[3].plot(time, S3)
    tmpax[3].set_ylabel("S" + str(3))
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))

    # tmpax[0].plot(time[1800::2], S00[1800::2], label=strlabel)
    # tmpax[0].set_ylabel("S" + str(0))
    # tmpax[0].legend(loc='upper right')
    # # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    # tmpax[1].plot(time[1800::2], S1)
    # tmpax[1].set_ylabel("S" + str(1))
    # # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    # tmpax[2].plot(time[1800::2], S2)
    # tmpax[2].set_ylabel("S" + str(2))
    # # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    # tmpax[3].plot(time[1800::2], S3)
    # tmpax[3].set_ylabel("S" + str(3))
    # # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))
    '''
    tmpax[0].plot(time[720:720+ndata], S0[720:720+ndata])
    tmpax[0].set_ylabel("S" + str(0))
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[1].plot(time[720:720+ndata], S1[720:720+ndata])
    tmpax[1].set_ylabel("S" + str(1))
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[2].plot(time[720:720+ndata], S2[720:720+ndata])
    tmpax[2].set_ylabel("S" + str(2))
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[3].plot(time[720:720+ndata], S3[720:720+ndata])
    tmpax[3].set_ylabel("S" + str(3))
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))
    '''
    tmpax[3].set_xlabel("Time (s)")
    tmpfig.align_ylabels()
    '''
    max_diff_S[nn][0] = S1.max() - S1.min()
    mean_S[nn][0] = (S1.max() + S1.min()) /2
    max_diff_S[nn][1] = S2.max() - S2.min()
    mean_S[nn][1] = (S2.max() + S2.min()) /2
    max_diff_S[nn][2] = S3.max() - S3.min()
    mean_S[nn][2] = (S3.max() + S3.min()) /2
    '''


custom_lines = [Line2D([0], [0], color=cstm_color[0], lw=4),
                Line2D([0], [0], color=cstm_color[1], lw=4),
                Line2D([0], [0], color=cstm_color[2], lw=4),
                Line2D([0], [0], color=cstm_color[3], lw=4)]

#fig2[0].legend(custom_lines, ['Pol.1', 'Pol.1+SOP Controller(w/o FB)', 'Pol.1+SOP Controller(w FB)'], loc='right')
#fig2[0].legend(custom_lines, ['Pol.1', 'Pol.1+Manual Controller', 'Pol.2+Manual Controller', 'Pol.1+Pol.2+ Manual Controller'], loc='right')
#fig2[0].legend(custom_lines, ['Measurement2', 'Measurement1'], loc='right')
fig2[0].legend(custom_lines, ['w/o polarizer', 'with polarizer poor alignment', 'with polarizer good alignment'], loc='right')
'''
for nn in range(3):
    delta = max_diff_S[:, nn].max()

    ax1[nn+1].set(ylim=(mean_S[0][nn] - delta / 1.9, mean_S[0][nn] + delta/1.9))
    ax2[nn+1].set(ylim=(mean_S[1][nn] - delta / 1.9, mean_S[1][nn] + delta/1.9))

fig3, ax3 = plt.subplots(figsize=(5, 4))

ax3.plot(frequency, diff_azi_V * 180 / pi, label="azimuth (deg)", marker="o")
ax3.plot(frequency, diff_ellip_V * 180 / pi, label="ellipticity (deg)", marker="v")
# label=r'$\theta$'
ax3.plot(frequency, sqrt(diff_azi_V ** 2 + diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)",
         marker="^")
ax3.xaxis.set_major_locator(MaxNLocator(5))

ax3.legend(loc="upper right")
ax3.set_xlabel("Vibration frequency (Hz)")
ax3.set_ylabel("Angle change (deg)")
plt.subplots_adjust(left=0.125, bottom=0.14, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
'''


plt.show()



