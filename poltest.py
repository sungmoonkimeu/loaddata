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


import os

# os.chdir('C:/Users/Iter/PycharmProjects/loaddata')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')
# os.chdir('C:/Users/SMK/PycharmProjects/loaddata/venv/')

#fig, ax = plt.subplots(figsize=(6, 5))
E1 = Jones_vector('Output_1')
E2 = Jones_vector('Output_2')

E1.linear_light(amplitude=1)
phase_m = arange(0, 2*pi, 0.01)
E2.linear_light(amplitude=0.02, azimuth=pi/2,  global_phase=phase_m)

E3 = E1+E2
E3.normalize()

Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

Sv.from_Jones(E3)
fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
frequency = arange(1,7,1)
diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))

for nn in range(len(file_list)):
    fig, ax = plt.subplots(4, figsize=(6, 5))
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
    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])

    azi_V = Out.parameters.azimuth()
    ellip_V = Out.parameters.ellipticity_angle()
    diff_azi_V[nn] = azi_V[0:1000].max() - azi_V[0:1000].min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()

    ax[0].plot(time, S0)
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[1].plot(time, S1)
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[2].plot(time, S2)
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[3].plot(time, S3)
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Stokes parameter")
    ax[0].set_title(file_list[nn])
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


#ax[3].set(xlim=(0, 2000), ylim=(0,1))

fig3, ax3 = plt.subplots(figsize=(6, 5))
#plt.rc('text', usetex=True)
#r'$\phi$'
ax3.scatter(frequency, diff_azi_V*180/pi, label="azimuth (deg)")
ax3.scatter(frequency, diff_ellip_V*180/pi, label="ellipticity (deg)")
#label=r'$\theta$'
ax3.scatter(frequency, sqrt(diff_azi_V**2 + diff_ellip_V**2)*180/pi, label="sqrt(azimuth^2 + ellipticity^2)")
#label=r'sqrt(\phi + \theta)')
ax3.legend(loc="best")
ax3.set_xlabel("Input pol. state")
ax3.set_ylabel("Angle change (deg)")
ax3.set(xlim=(0.5, 6.5), ylim=(0, 5))

my_xticks = ['LHP','LVP','L45P','L135P','LCP','RCP']
plt.xticks(frequency,my_xticks)



plt.show()

