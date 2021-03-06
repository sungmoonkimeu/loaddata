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
from py_pol.jones_matrix import Jones_matrix
from py_pol.mueller import Mueller
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

def basistonormal(S):
    #S = create_Stokes('Output')
    S2 = create_Stokes('cal')
    J1 = Jones_matrix('Random element')
    J2 = Jones_matrix('Random element')
    M = Mueller('cal')

    a = S.parameters.matrix()[1:]  # convert 4x1 Stokes vectors to 3x1 cartesian vectors

    ''' 
    # ?????? ??????
    mean_a = np.array([a[0, :].sum(), a[1, :].sum(), a[2, :].sum()])
    mean_a = mean_a / (np.linalg.norm(mean_a))
    print(mean_a)
    # ?????? ????????? ?????? ??? ????????? ??????
    dist_a_mean_a = np.linalg.norm(a.T - mean_a, axis=1)
    # ??????????????? ?????? ????????? ?????? --> ?????? ?????? ?
    std_a = a[:, np.argmin(dist_a_mean_a)]

    # ?????? ?????? ??? ????????? ?????? ??????
    diff_a = a.T - std_a

    # ?????? ????????? ????????? ????????? ????????? ?????? ?????? ??????
    cross_a = np.cross(diff_a[0], diff_a)

    # filtering too small vectors
    cross_a2 = cross_a[np.linalg.norm(cross_a, axis=1) > np.linalg.norm(cross_a, axis=1).mean() / 10]
    # ?????? ?????? vector ?????? ????????????
    cross_an = cross_a2.T / np.linalg.norm(cross_a2, axis=1)
    # Normalize
    cross_an_abs = cross_an * abs(cross_an.sum(axis=0)) / cross_an.sum(axis=0)
    # average after summation whole vectors
    c = cross_an_abs.sum(axis=1) / np.linalg.norm(cross_an_abs.sum(axis=1))
    '''

    # ?????? ????????? ??????
    c = a[...,0]
    print(c)

    #print("new c", c)
    #fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1, )

    z = [0, 0, 1]
    y = [0, 1, 0]
    x = [1, 0, 0]

    th_x = np.arccos(np.dot(x, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_y = np.arccos(np.dot(y, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_z = np.arccos(np.dot(z, c))

    #print("x=", th_x * 180 / pi, "y=", th_y * 180 / pi, "z=", th_z * 180 / pi)

    th = -th_y
    if th_x > pi / 2:
        th = th_y
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R ?????? rotation

    th = 0
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + ?????? rotation

    th = pi/2-th_z
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H ?????? rotation


    TT = R45.T @ Rh.T @ Rr.T @ a
    zT = ones(np.shape(TT)[1])

    Sp = np.vstack((zT, TT))
    S.from_matrix(Sp)
    return S


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
#path_dir = os.getcwd() + '//Data_Vib_4(Hibi_loosen)//Polscan_1504_edited'
#path_dir = os.getcwd() + '//Data_Vib_4(Hibi_fasten)//Pol_Scan_Tighten_1504_edited'
#path_dir = os.getcwd() + '//Data_Vib_4(Lobi)/Pol_Scan_Lobi_1_1504_edited'
path_dir = os.getcwd() + '//Data_Vib_4(Lobi)/Pol_Scan_3basis_1504_edited'


file_list = os.listdir(path_dir)

#fig, ax = plt.subplots(figsize=(6, 5))
Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
#ang_SOP = arange(0, 361, 5)
ang_SOP = arange(0, 361, 180)

diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))
max_azi_V = np.ones(len(file_list))
min_azi_V = np.ones(len(file_list))
alpha= np.ones(len(file_list))

# file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[0][2:]))
for nn in range(len(file_list)):

    fn2 = path_dir + "//" + file_list[nn]

    print(fn2)
    count = 0
    cstm_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

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
    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 8])

    Out = basistonormal(Out)

    azi_V = Out.parameters.azimuth()

    #print(azi_V[0])
    ellip_V = Out.parameters.ellipticity_angle()
    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    if diff_azi_V[nn] > pi/2:
        tmp = azi_V > pi/2
        azi_V[tmp] = azi_V[tmp]-pi
        diff_azi_V[nn] = azi_V.max() - azi_V.min()

    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()
    max_azi_V[nn] = azi_V.max()/cos(ellip_V[0])
    min_azi_V[nn] = azi_V.min()/cos(ellip_V[0])

    alpha[nn] = sqrt(diff_azi_V[nn]**2 + diff_ellip_V[nn]**2)


    if alpha[nn] > 0.1:
        print(fn2)
        print(diff_azi_V[nn])
        print(diff_ellip_V[nn])
        print(cos(ellip_V[0]))

    if nn == 0 or nn == 1 or nn==2 :
        fig, ax = plt.subplots(4, figsize=(6, 5))
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
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(ang_SOP,alpha*180/pi)
ax.plot(ang_SOP,diff_azi_V*180/pi)
ax.plot(ang_SOP,diff_ellip_V*180/pi)


'''
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.scatter(frequency, diff_azi_V*180/pi, label="azimuth (deg)")
ax3.scatter(frequency, diff_ellip_V*180/pi, label="ellipticity (deg)")
ax3.scatter(frequency, sqrt(diff_azi_V**2 + diff_ellip_V**2)*180/pi, label="sqrt(azimuth^2 + ellipticity^2)")
ax3.legend(loc="best")
ax3.set_xlabel("Input pol. state")
ax3.set_ylabel("Angle change (deg)")
ax3.set(xlim=(0.5, 6.5), ylim=(0, 5))

my_xticks = ['LHP', 'LVP', 'L45P', 'L135P', 'LCP', 'RCP']
plt.xticks(frequency, my_xticks)

'''
plt.show()

