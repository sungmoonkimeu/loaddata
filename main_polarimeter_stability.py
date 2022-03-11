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

#foldername = '//Laser stability test_2nd'
#foldername = '//Stability_ManualPC'
foldername = '//Laser_stability_test_cascadedpol'

path_dir = os.getcwd() + foldername + '_edited'

file_list = os.listdir(path_dir)

plt.subplots_adjust(left=0.14, bottom=0.112, right=0.93, top=0.93, wspace=0.2, hspace=0)

Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
frequency = arange(10, 11,1)
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
    count = 0
    cstm_color = ['c', 'm', 'y', 'k', 'r']

    #    fn2 = path_dir + "//10Hz_edited.txt"
    data = pd.read_table(fn2, delimiter=r"\s+")
    #time = pd.to_numeric(data['Time(ms)']) / 10000
    #time = np.arange(0, 3600, 1)/3600

    S0 = pd.to_numeric(data['S0(mW)'])
    S1 = pd.to_numeric(data['S1'])
    S2 = pd.to_numeric(data['S2'])
    S3 = pd.to_numeric(data['S3'])
    time = np.arange(0, len(S0), 1) / 720

    Sn = np.ones((len(S0)))
    SS = np.vstack((Sn[720:], S1[720:], S2[720:], S3[720:]))
    Out = Sv.from_matrix(SS.T)

    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])

    azi_V = Out.parameters.azimuth()
    print(azi_V)
    ellip_V = Out.parameters.ellipticity_angle()
    for i_n, i_v in enumerate(azi_V):
        if i_v > pi/2:
            azi_V[i_n] = i_v - pi
    for i_n, i_v in enumerate(ellip_V):
        if i_v > pi / 2:
            ellip_V[i_n] = i_v - pi
        if i_v > 0.8:
            print(i_n)
            ellip_V[i_n] = 0

    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()
    print(azi_V)

    if nn == 0:
        tmpax = ax1
        tmpfig = fig_1
    else:
        tmpax = ax2
        tmpfig = fig_2

    tmpax[0].plot(time[720:], S0[720:])
    tmpax[0].set_ylabel("S" + str(0))
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[1].plot(time[720:], S1[720:])
    tmpax[1].set_ylabel("S" + str(1))
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[2].plot(time[720:], S2[720:])
    tmpax[2].set_ylabel("S" + str(2))
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    tmpax[3].plot(time[720:], S3[720:])
    tmpax[3].set_ylabel("S" + str(3))
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))

    tmpax[3].set_xlabel("Time (h)")
    tmpfig.align_ylabels()

    max_diff_S[nn][0] = S1.max() - S1.min()
    mean_S[nn][0] = (S1.max() + S1.min()) /2
    max_diff_S[nn][1] = S2.max() - S2.min()
    mean_S[nn][1] = (S2.max() + S2.min()) /2
    max_diff_S[nn][2] = S3.max() - S3.min()
    mean_S[nn][2] = (S3.max() + S3.min()) /2


for nn in range(3):
    delta = max_diff_S[:, nn].max()

    ax1[nn+1].set(ylim=(mean_S[0][nn] - delta / 1.9, mean_S[0][nn] + delta/1.9))
    ax2[nn+1].set(ylim=(mean_S[1][nn] - delta / 1.9, mean_S[1][nn] + delta/1.9))


#ax[3].set(xlim=(0, 2000), ylim=(0,1))

fig3, ax3 = plt.subplots(figsize=(5, 4))
#plt.rc('text', usetex=True)
#r'$\phi$'

ax3.plot(frequency, diff_azi_V * 180 / pi, label="azimuth (deg)", marker="o")
ax3.plot(frequency, diff_ellip_V * 180 / pi, label="ellipticity (deg)", marker="v")
# label=r'$\theta$'
ax3.plot(frequency, sqrt(diff_azi_V ** 2 + diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)",
         marker="^")
ax3.xaxis.set_major_locator(MaxNLocator(5))

#ax3.plot(frequency, new_diff_azi_V * 180 / pi, label="azimuth (deg)2", marker="x")
#ax3.plot(frequency, new_diff_ellip_V * 180 / pi, label="ellipticity (deg)2", marker="x")
# label=r'$\theta$'
#ax3.plot(frequency, sqrt(new_diff_azi_V ** 2 + new_diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)2",
#         marker="x")

#label=r'sqrt(\phi + \theta)')
ax3.legend(loc="upper right")
ax3.set_xlabel("Vibration frequency (Hz)")
ax3.set_ylabel("Angle change (deg)")
#ax3.set(xlim=(10, 30), ylim=(0, 1.7))
plt.subplots_adjust(left=0.125, bottom=0.14, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

plt.show()


