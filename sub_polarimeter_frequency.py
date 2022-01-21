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

foldername = 'Const_disp_Polarimeter2'

path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
file_list = os.listdir(path_dir)

fig, ax = plt.subplots(4, figsize=(6, 5))
Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

frequency = arange(10, 31, 1)

diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))

fig4, ax4 = plt.subplots(2, figsize=(5, 4))

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
    # SS = np.vstack((S0, S1, S2, S3))
    SS = np.vstack((Sn, S1, S2, S3))

    Out = Sv.from_matrix(SS.T)

    azi_V = Out.parameters.azimuth()
    ellip_V = Out.parameters.ellipticity_angle()
    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()

    '''
    ax[0].plot(time, S0)
    ax[0].set(xlim=(0, 0.5))
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[1].plot(time, S1)
    ax[1].set(xlim=(0, 0.5))
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[2].plot(time, S2)
    ax[2].set(xlim=(0, 0.5))
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[3].plot(time, S3,label=str(nn+10)+'Hz')
    ax[3].set(xlim=(0, 0.5))
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))
    '''
    ax4[0].plot(time, azi_V)
    ax4[1].plot(time, ellip_V, label=str(nn+10)+'Hz')

'''
ax[3].set_xlabel("Time (s)")
ax[0].set_title("Stokes parameter")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 2.2))

'''
ax4[1].set_xlabel("Time (s)")
ax4[0].set_ylabel("Ellipticity angle (deg)")
ax4[1].set_ylabel("Azimuth angle (deg)")
plt.legend(loc="center left", bbox_to_anchor=(1.04, 1.2))

#ax[3].set(xlim=(0, 2000), ylim=(0,1))

fig3, ax3 = plt.subplots(figsize=(5, 4))
#plt.rc('text', usetex=True)
#r'$\phi$'

ax3.plot(frequency, diff_azi_V * 180 / pi, label="azimuth (deg)", marker="o")
ax3.plot(frequency, diff_ellip_V * 180 / pi, label="ellipticity (deg)", marker="v")
# label=r'$\theta$'
ax3.plot(frequency, sqrt(diff_azi_V ** 2 + diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)",
         marker="^")

#label=r'sqrt(\phi + \theta)')
ax3.legend(loc="upper right")
ax3.set_xlabel("Vibration frequency (Hz)")
ax3.set_ylabel("Angle change (deg)")
ax3.set(xlim=(10, 30), ylim=(0, 1.7))
plt.subplots_adjust(left=0.125, bottom=0.14, right=0.9, top=0.9, wspace=0.2, hspace=0.2)





plt.show()

