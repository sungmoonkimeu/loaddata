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
from py_pol.mueller import Mueller
from py_pol.jones_matrix import Jones_matrix


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
    # 평균 벡터
    mean_a = np.array([a[0, :].sum(), a[1, :].sum(), a[2, :].sum()])
    mean_a = mean_a / (np.linalg.norm(mean_a))
    print(mean_a)
    # 평균 벡터와 모든 점 사이의 거리
    dist_a_mean_a = np.linalg.norm(a.T - mean_a, axis=1)
    # 평균벡터와 가장 가까운 벡터 --> 대표 벡터 ?
    std_a = a[:, np.argmin(dist_a_mean_a)]

    # 대표 벡터 와 나머지 벡터 연결
    diff_a = a.T - std_a

    # 대표 벡터와 나머지 벡터가 이루는 벡터 끼리 외적
    cross_a = np.cross(diff_a[0], diff_a)

    # filtering too small vectors
    cross_a2 = cross_a[np.linalg.norm(cross_a, axis=1) > np.linalg.norm(cross_a, axis=1).mean() / 10]
    # 반대 방향 vector 같은 방향으로
    cross_an = cross_a2.T / np.linalg.norm(cross_a2, axis=1)
    # Normalize
    cross_an_abs = cross_an * abs(cross_an.sum(axis=0)) / cross_an.sum(axis=0)
    # average after summation whole vectors
    c = cross_an_abs.sum(axis=1) / np.linalg.norm(cross_an_abs.sum(axis=1))
    '''

    # 그냥 첫번쨰 벡터
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
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

    th = 0
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + 기준 rotation

    th = pi/2-th_z
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation


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

#foldername = 'Const_appl_vol_Polarimeter'
#foldername = '0_RHC_losen'
#foldername = '2_RHC'
#foldername = 'Const_disp_Polarimeter2'
V_foldername = ['2_LP0_loosen', '2_LP45', '8_RHC_loosen']
#V_foldername = ['1_LP0', '1_LP45', '1_RHC_fasten']
#V_foldername = ['Const_volt_LP90_Polarimeter', 'Const_volt_LP45_Polarimeter', 'Const_volt_RHC_Polarimeter']
V_label = ['LP0', 'LP45', 'RHC']
V_marker = ['^', 'o', 'x']
fig3, ax3 = plt.subplots(figsize=(5, 4))
for n_iter, foldername in enumerate(V_foldername):

    #path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
    path_dir = os.getcwd() + '//Data_Vib_3_(Hibi_loosen_fasten)//' + foldername + '_edited'

    #path_dir = os.getcwd() + '//Data_Vib_2_(Hibi_losen_fasten)//' + foldername + '_edited'
    #path_dir = os.getcwd() + '//Data_Vib_3_(Hibi_loosen_fasten)//' + foldername + '_edited'
    file_list = os.listdir(path_dir)

    fig, ax = plt.subplots(4, figsize=(6, 5))
    plt.subplots_adjust(left=0.14, bottom=0.112, right=0.93, top=0.93, wspace=0.2, hspace=0)

    Ev = Jones_vector('Output_J')
    Sv = create_Stokes('Output_S')
    Out = create_Stokes('Output_S2')

    fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')
    frequency = arange(10, 31, 1)
    #frequency = np.array([30,31])

    diff_azi_V = np.ones(len(file_list))
    diff_ellip_V = np.ones(len(file_list))
    cstm_color = ['k', 'r', 'b', 'c', 'y', 'm']

    for nn in range(len(file_list)):
        fn2 = path_dir + "//" + file_list[nn]
        count = 0

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

        Out = basistonormal(Out)

        draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])

        azi_V = Out.parameters.azimuth()
        ellip_V = Out.parameters.ellipticity_angle()
        diff_azi_V[nn] = azi_V.max() - azi_V.min()
        diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()

        if nn == 0 or nn == len(file_list)-2:
            ax[0].plot(time, S0)
            ax[1].plot(time, S1)
            ax[2].plot(time, S2)
            ax[3].plot(time, S3)

    for nn in range(len(ax)):
        ax[nn].set_ylabel("S"+str(nn))
    ax[3].set_xlabel("Time (s)")
    fig.align_ylabels()

    #plt.rc('text', usetex=True)
    #r'$\phi$'

    ax3.plot(frequency, sqrt(diff_azi_V ** 2 + diff_ellip_V ** 2) * 180 / pi, label=V_label[n_iter],
             marker=V_marker[n_iter], color=cstm_color[n_iter % 4])
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax3.set(xlim=(9, 31), ylim=(0, 2))

    #label=r'sqrt(\phi + \theta)')
    ax3.legend(loc="upper left")
    ax3.set_xlabel("Vibration frequency (Hz)")
    ax3.set_ylabel("SOP change (deg)")
    #ax3.set(xlim=(10, 30), ylim=(0, 1.7))
    plt.subplots_adjust(left=0.152, bottom=0.133, right=0.917, top=0.89, wspace=0.2, hspace=0.2)

plt.show()


