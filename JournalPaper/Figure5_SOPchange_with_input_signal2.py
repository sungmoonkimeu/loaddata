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

    y0 = (max(signal0) - min(signal0))      # peak-peak voltage
    y1 = (max(signal1) - min(signal1)) * 10  # Acceleration
    y2 = y0 / (frequency ** 2) * 1000       # displacement

    return np.array([y0, y1, y2])


if __name__ == '__main__':

    path = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//'
    folder1 = 'Const_acc_OSC2_edited//'
    folder2 = 'Const_acc_Polarimeter_edited//'

    folder3 = 'Const_disp_OSC2_edited//'
    folder4 = 'Const_disp_Polarimeter2_edited//'



    freq = np.arange(10, 31, 1)

    displacement = np.zeros(len(freq))
    acceleration = np.zeros(len(freq))
    alpha1 = np.zeros(len(freq))
    alpha2 = np.zeros(len(freq))

    for nn in freq:
        # Const displacement
        filename1 = path + folder1 + 'scope_' + str(nn-10) + '_edited.txt'

        y_array = read_shakersignal(filename1, nn)
        displacement[nn-10] = y_array[2]

        filename2 = path + folder2 + str(nn) + 'Hz_1_edited.txt'
        S = read_SOP(filename2)
        delta_azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        delta_ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha1[nn-10] = sqrt(delta_azi**2 + delta_ellip**2) * 180/pi

        # Const acceleration
        filename3 = path + folder3 + 'scope_' + str(nn-10) + '_edited.txt'
        #print(filename)
        y_array = read_shakersignal(filename3, nn)
        acceleration[nn - 10] = y_array[1]

        filename4 = path + folder4 + str(nn) + 'Hz_edited.txt'
        S = read_SOP(filename4)
        S = basistonormal(S)
        delta_azi = S.parameters.azimuth().max() - S.parameters.azimuth().min()
        delta_ellip = S.parameters.ellipticity_angle().max() - S.parameters.ellipticity_angle().min()
        alpha2[nn-10] = sqrt(delta_azi**2 + delta_ellip**2) * 180/pi

    fig, ax = plt.subplots(1, 2, figsize=(23/2.54, 8/2.54))
    fig.set_dpi(91.79)  # DPI of My office monitor
    plt.subplots_adjust(left=0.088, bottom=0.155, right=0.91, top=0.93, wspace=0.593, hspace=0.502)
    plt.subplots_adjust(bottom=0.155)

    ms = 4
    ax[0].plot(freq, alpha1, lw='1', label="Max. SOP change", marker='o', color='k', markersize=ms)
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Max. SOP change (deg)')
    #ax[0].set_title('Input signal')
    ax[0].set(xlim=(10, 30), ylim=(0, 2))
    lns01 = ax[0].lines
    ax0 = ax[0].twinx()
    lns02 = ax0.plot(freq, displacement, lw='1', label="Displacement", marker='x', color='r', markersize=ms)
    ax0.set_ylabel('Displacement (mm)')
    ax0.set(ylim=(0, 180))

    # added these three lines
    lns = lns01 + lns02
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc=0)

    ax[1].plot(freq, alpha2, lw='1', label="Max. SOP change", marker='o', color='k', markersize=ms)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Max. SOP change (deg)')
    #ax[1].set_title('Measured signal')
    ax[1].set(xlim=(10, 30), ylim=(0, 2))
    lns11 = ax[1].lines
    ax1 = ax[1].twinx()
    lns12 = ax1.plot(freq, acceleration, lw='1', label="Acceleration", marker='^', color='b', markersize=ms)
    ax1.set_ylabel('Acceleration (g)')
    ax1.set(ylim=(0, 40))

    # added these three lines
    lns = lns11 + lns12
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=0)

    fig.align_ylabels()
    #fig.savefig('Constant_Acceleration.png')
    #fig.savefig('Constant_Displacement.png')

plt.show()

