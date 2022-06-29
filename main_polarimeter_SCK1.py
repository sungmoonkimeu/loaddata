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
import matplotlib.pylab as pl
from matplotlib.colors import rgb2hex

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

from tkinter import Tk, filedialog
import os
import sys
print(os.getcwd())
print(os.getcwd()+'\My_library')
sys.path.append(os.getcwd()+'\My_library')

import plotly.graph_objects as go
import draw_poincare_plotly as PS


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


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def basistonormal(S):
    # S = create_Stokes('Output')
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
    c = a[..., 0]
    print(c)

    # print("new c", c)
    # fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1, )

    z = [0, 0, 1]
    y = [0, 1, 0]
    x = [1, 0, 0]

    th_x = np.arccos(np.dot(x, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_y = np.arccos(np.dot(y, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_z = np.arccos(np.dot(z, c))

    # print("x=", th_x * 180 / pi, "y=", th_y * 180 / pi, "z=", th_z * 180 / pi)

    th = -th_y
    if th_x > pi / 2:
        th = th_y
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

    th = 0
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + 기준 rotation

    th = pi / 2 - th_z
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation

    TT = R45.T @ Rh.T @ Rr.T @ a
    zT = ones(np.shape(TT)[1])

    Sp = np.vstack((zT, TT))
    S.from_matrix(Sp)
    return S


def PS3(shot):
    '''
    plot Poincare Sphere, ver. 20/03/2020
    return:
    ax, fig
    '''
    fig = plt.figure(figsize=(6, 6))
    #    plt.figure(constrained_layout=True)
    ax = Axes3D(fig)
    # white panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # no ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # no panes
    ax.set_axis_off()

    # plot greed
    u = np.linspace(0, 2 * np.pi, 61)  # azimuth
    v = np.linspace(0, np.pi, 31)  # elevation
    sprad = 1
    x = sprad * np.outer(np.cos(u), np.sin(v))
    y = sprad * np.outer(np.sin(u), np.sin(v))
    z = sprad * np.outer(np.ones(np.size(u)), np.cos(v))


    ax.plot_surface(x, y, z,
                    color='w',  # (0.5, 0.5, 0.5, 0.0),
                    #edgecolor='k',
                    edgecolor=(3/256, 3/256, 3/256),
                    linestyle=(0, (5, 5)),
                    rstride=3, cstride=3,
                    linewidth=.5, alpha=0.8, shade=
                    0)


    # main circles
    #ax.plot(np.sin(u), np.cos(u), np.zeros_like(u), 'g-.', linewidth=0.75)  # equator
    #    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), 'b-', linewidth=0.5)
    #    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), 'b-', linewidth=0.5)

    # axes and captions
    amp = 1.3 * sprad
    ax.plot([-amp, amp], [0, 0], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=1)
    ax.plot([0, 0], [-amp, amp], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=1)
    ax.plot([0, 0], [0, 0], [-amp, amp], 'k-.', lw=1, alpha=0.5, zorder=1)

    distance = 1.3 * sprad
    ax.text(distance, 0, 0, '$S_1$', fontsize=18)
    ax.text(0, -distance, 0, '$S_2$', fontsize=18)
    ax.text(0, 0, -distance, '$S_3$', fontsize=18)

    # points
    px = [1, -1, 0, 0, 0, 0]
    py = [0, 0, 1, -1, 0, 0]
    pz = [0, 0, 0, 0, 1, -1]

    ax.plot(px, py, pz,
            color='black', marker='o', markersize=4, alpha=1, linewidth=0)
    #

    max_size = 1.05 * sprad
    ax.set_xlim(-max_size, max_size)
    ax.set_ylim(-max_size, max_size)
    ax.set_zlim(-max_size, max_size)

    #    plt.tight_layout()            #not compatible
    #ax.view_init(elev=-21, azim=-54)
    ax.view_init(elev=-160, azim=110)
    #    ax.view_init(elev=0/np.pi, azim=0/np.pi)

    #    ax.set_title(label = shot, loc='left', pad=10)
    ax.set_title(label="  " + shot, loc='left', pad=-10, fontsize=8)

    #    ax.legend()

    ax.set_box_aspect([1, 1, 1])

    return ax, fig


if __name__ =='__main__':

    # figure setting
    plt.close("all")
    plt_fmt, plt_res = '.png', 330  # 330 is max in Word'16
    plt.rcParams["axes.titlepad"] = 5  # offset for the fig title
    #plt.rcParams["figure.autolayout"] = True  # tight_layout
    #  plt.rcParams['figure.constrained_layout.use'] = True  # fit legends in fig window
    fsize = 9
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
    plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('legend', fontsize=fsize-1)  # f-size legend
    plt.rc('axes', titlesize=11)  # f-size of the axes title (??)
    plt.rc('figure', titlesize=11)  # f-size of the figure title

    # SOP change
    fig, ax = plt.subplots(figsize=(8.5/2.54, 7/2.54))
    plt.subplots_adjust(left=0.2, bottom=0.22)

    # fig, ax = plt.subplots(figsize=(6, 5))
    ax2, fig2 = PS3('0')
    fig3 = PS.PS5()

    for mm in range(1):
        # Folder select
        cwd = os.getcwd()
        rootdirectory = os.path.dirname(os.getcwd())
        root = Tk()  # pointing root to Tk() to use it as Tk() in program.
        root.withdraw()  # Hides small tkinter window.
        root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite of selection.
        path_dir = filedialog.askdirectory(initialdir=rootdirectory)  # Returns opened path as str
        file_list = os.listdir(path_dir)

        Ev = Jones_vector('Output_J')
        Sv = create_Stokes('Output_S')
        Out = create_Stokes('Output_S2')

        # colors = pl.cm.brg(np.linspace(0, 1, len(file_list)))
        # brg = matplotlib_to_plotly(pl.cm.brg, len(file_list))
        # hsv = matplotlib_to_plotly(pl.cm.hsv, len(file_list))

        nn = 0
        fn2 = path_dir + "//" + file_list[nn]
        print(fn2)
        count = 0
        cstm_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        #    fn2 = path_dir + "//10Hz_edited.txt"
        pddata = pd.read_table(fn2, delimiter=",")
        data = (pddata.to_numpy()).T
        time = pd.to_numeric(data[0])
        S0 = pd.to_numeric(data[1])
        S1 = pd.to_numeric(data[2])
        S2 = pd.to_numeric(data[3])
        S3 = pd.to_numeric(data[4])

        Sn = np.ones((len(S0)))

        SS = np.vstack((Sn, S1, S2, S3))
        Out = Sv.from_matrix(SS.T)

        # draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 8])

        # Out = basistonormal(Out)
        # draw_stokes_points(fig3[0], Out, kind='line', color_line=cstm_color[nn % 8])
        ax2.plot(S1, S2, S3, color='c', marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=3)

        # fig3.add_scatter3d(x=S1[::10], y=S2[::10], z=S3[::10], mode="markers",
        #                    marker=dict(size=3, opacity=1, color=rgb2hex(colors[nn])), name='F1')


        if nn == 0 or nn == 5 or nn == 9 or nn == 14:
            fig4, ax4 = plt.subplots(4, figsize=(6, 5))
            ax4[0].plot(time, S0)
            # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
            ax4[1].plot(time, S1)
            # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
            ax4[2].plot(time, S2)
            # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
            ax4[3].plot(time, S3)
            # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))
            ax4[3].set_xlabel("Time (s)")
            ax4[3].set_ylabel("Stokes parameter")
            ax4[0].set_title(file_list[nn])

    plt.show()

