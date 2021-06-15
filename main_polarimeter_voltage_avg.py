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

foldername = 'Const_Freq_Polarimeter'

path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
file_list = os.listdir(path_dir)

fig, ax = plt.subplots(4, figsize=(6, 5))
Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
applied_V = arange(0.4, 4.2, 0.2)

diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))
new_diff_azi_V = np.ones(len(file_list))
new_diff_ellip_V = np.ones(len(file_list))

mean_azi_V = np.ones(len(file_list))
err_azi_V = np.ones(len(file_list))
mean_ellip_V = np.ones(len(file_list))
err_ellip_V = np.ones(len(file_list))
mean_nor_SOP_V = np.ones(len(file_list))
err_nor_SOP_V = np.ones(len(file_list))



for nn in range(len(file_list)):
    fn2 = path_dir + "//" + str(400+nn*200) + "mV_edited.txt"
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


    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])

    azi_V = Out.parameters.azimuth()
    ellip_V = Out.parameters.ellipticity_angle()
    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()
    nor_SOP_V = sqrt((azi_V - azi_V.mean())**2 + (ellip_V - ellip_V.mean())**2)
    nor_SOP_V = nor_SOP_V-nor_SOP_V.mean()

    mean_azi_V[nn] = azi_V.mean()
    err_azi_V[nn] = azi_V.std()
    mean_ellip_V[nn] = ellip_V.mean()
    err_ellip_V[nn] = ellip_V.std()
    mean_nor_SOP_V[nn] = nor_SOP_V.mean()
    err_nor_SOP_V[nn] = nor_SOP_V.std()

    nwindow = 10
    rS1 = S1.rolling(window=nwindow)
    rS2 = S2.rolling(window=nwindow)
    rS3 = S3.rolling(window=nwindow)

    new_S1 = rS1.mean()
    new_S2 = rS2.mean()
    new_S3 = rS3.mean()
    new_S1[0:nwindow] = new_S1[nwindow]
    new_S2[0:nwindow] = new_S2[nwindow]
    new_S3[0:nwindow] = new_S3[nwindow]

    new_SS = np.vstack((Sn, new_S1, new_S2, new_S3))
    new_Out = Sv.from_matrix(new_SS.T)

    new_azi_V = new_Out.parameters.azimuth()
    new_ellip_V = new_Out.parameters.ellipticity_angle()
    new_diff_azi_V[nn] = new_azi_V.max() - new_azi_V.min()
    new_diff_ellip_V[nn] = new_ellip_V.max() - new_ellip_V.min()

    ax[0].plot(time, S0)
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[1].plot(time, S1)
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[2].plot(time, S2)
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax[3].plot(time, S3)
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))

    ax[0].plot(time, S0)
    ax[1].plot(time, new_S1)
    ax[2].plot(time, new_S2)
    ax[3].plot(time, new_S3)


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

ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Stokes parameter")
#ax[3].set(xlim=(0, 2000), ylim=(0,1))

fig3, ax3 = plt.subplots(figsize=(5, 4))
#plt.rc('text', usetex=True)
#r'$\phi$'

ax3.plot(applied_V, diff_azi_V * 180 / pi, label="azimuth (deg)", marker="o")
ax3.plot(applied_V, diff_ellip_V * 180 / pi, label="ellipticity (deg)", marker="v")
# label=r'$\theta$'
ax3.plot(applied_V, sqrt(diff_azi_V ** 2 + diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)",
         marker="^")

ax3.plot(applied_V, new_diff_azi_V * 180 / pi, label="azimuth (deg)2", marker="x")
ax3.plot(applied_V, new_diff_ellip_V * 180 / pi, label="ellipticity (deg)2", marker="x")
# label=r'$\theta$'
ax3.plot(applied_V, sqrt(new_diff_azi_V ** 2 + new_diff_ellip_V ** 2) * 180 / pi, label="sqrt(azimuth^2 + ellipticity^2)2",
         marker="x")

#label=r'sqrt(\phi + \theta)')
ax3.legend(loc="upper right")
ax3.set_xlabel("Applied voltage (V)")
ax3.set_ylabel("Angle change (deg)")
#ax3.set(xlim=(10, 30), ylim=(0, 1.7))
plt.subplots_adjust(left=0.125, bottom=0.14, right=0.9, top=0.9, wspace=0.2, hspace=0.2)


fig3, ax3 = plt.subplots(figsize=(5, 4))
plt.subplots_adjust(left=0.157, bottom=0.11, right=0.955, top=0.886, wspace=0.2, hspace=0.2)
#ax3.errorbar(frequency, mean_azi_V * 180 / pi, yerr=0.25,
ax3.scatter(applied_V, mean_azi_V * 180 / pi, s=10, c='black', label="mean value", marker='o', zorder=100)

ax3.errorbar(applied_V, mean_azi_V * 180 / pi, yerr=0.25,
             label="uncertainty of device",  ls="None", c='black', ecolor='lightgray', elinewidth=3, capsize=6)
ax3.errorbar(applied_V, mean_azi_V * 180 / pi,  yerr=err_azi_V*180/pi,
             label="standard deviation", ls="None", c='black', ecolor='g', capsize=4, zorder=5)

ax3.legend(loc="upper right")
ax3.xaxis.set_major_locator(MaxNLocator(5))
ax3.set_xlabel("Applied voltage (V)")
ax3.set_ylabel("Azimuth (deg)")
ax3.set(xlim=(0, 4.2), ylim=(147.5, 148.5))

fig3, ax3 = plt.subplots(figsize=(5, 4))
plt.subplots_adjust(left=0.157, bottom=0.11, right=0.955, top=0.886, wspace=0.2, hspace=0.2)
ax3.scatter(applied_V, mean_ellip_V * 180 / pi, s=10, c='black', label="mean value", marker='o', zorder=100)
ax3.errorbar(applied_V, mean_ellip_V * 180 / pi, yerr=0.25,
             label="uncertainty of device", ls="None", ecolor='lightgray', elinewidth=3, capsize=6)
ax3.errorbar(applied_V, mean_ellip_V * 180 / pi, yerr=err_ellip_V*180/pi,
             label="standard deviation", ls="None", ecolor='g', capsize=4, zorder=5)
ax3.legend(loc="lower right")
ax3.xaxis.set_major_locator(MaxNLocator(5))
ax3.set_xlabel("Applied voltage (V)")
ax3.set_ylabel("Ellipticity (deg)")
ax3.set(xlim=(0, 4.2), ylim=(-17.7, -16.7))


fig3, ax3 = plt.subplots(figsize=(5, 4))
plt.subplots_adjust(left=0.157, bottom=0.11, right=0.955, top=0.886, wspace=0.2, hspace=0.2)

ax3.errorbar(applied_V, mean_nor_SOP_V * 180 / pi, yerr=0.25,
             label="Device's uncertainty", ecolor='lightgray', elinewidth=3, capsize=6)
ax3.errorbar(applied_V, mean_nor_SOP_V * 180 / pi, yerr=err_nor_SOP_V*180/pi,
             label="STD", fmt='o', ecolor='g', capsize=4)
ax3.legend(loc="upper right")
ax3.xaxis.set_major_locator(MaxNLocator(5))
ax3.set_xlabel("Applied voltage (V)")
ax3.set_ylabel("Normalized SOP uncertainty (deg)")


plt.show()

