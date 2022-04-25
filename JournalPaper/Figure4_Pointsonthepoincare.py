# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:50:03 2022

@author: agoussar
"""

import pickle
import scipy.io
import scipy.optimize
from scipy import stats
from cycler import cycler
import numpy as np
from numpy import pi, arccos, cos, sin
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time
from itertools import chain
import warnings
import os
import pandas as pd

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
from py_pol.mueller import Mueller
from py_pol.jones_matrix import Jones_matrix

# Switching OS folder
path2 = 'C:/Users/Iter/PycharmProjects/loaddata'
path1 = 'C:/Users/SMK/PycharmProjects/loaddata'
def switch_osfolder():
    try:
        if os.path.exists(path1):
            os.chdir(path1)
        else:
            os.chdir(path2)
    except OSError:
        print('Error: Changing OS directory')
switch_osfolder()

plt.style.use('seaborn-paper')
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", message="This figure includes Axes")
warnings.filterwarnings("ignore", message="This figure was using")
warnings.filterwarnings("ignore", message="Calling figure.constrained_layout")


def basistonormal(S):

    #a = S.parameters.matrix()[1:]  # convert 4x1 Stokes vectors to 3x1 cartesian vectors
    a = S  # convert 4x1 Stokes vectors to 3x1 cartesian vectors

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

    th = pi/2-th_z+pi
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation

    TT = R45.T @ Rh.T @ Rr.T @ a

    return TT


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

def main():
    #
    # plot parameters
    #
    Sv = create_Stokes('Output_S')

    plt.close("all")
    plt_fmt, plt_res = '.png', 330  # 330 is max in Word'16
    plt.rcParams["axes.titlepad"] = 20  # offset for the fig title
    #  plt.rcParams['figure.constrained_layout.use'] = True  # fit legends in fig window
    fsize = 11
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.sans-serif'] = "calibri"
    plt.rcParams['font.family'] = "sans-serif"

    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
    plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('legend', fontsize=fsize)  # f-size legend
    plt.rc('axes', titlesize=6)  # f-size of the axes title (??)
    plt.rc('figure', titlesize=6)  # f-size of the figure title
    #

    foldername = 'Const_acc_Polarimeter'

    path_dir = 'Data_Vib_1_(Oscillo_Polarimeter)//' + foldername + '_edited'
    file_list = os.listdir(path_dir)

    Ev = Jones_vector('Output_J')
    Sv = create_Stokes('Output_S')
    Out = create_Stokes('Output_S2')

    #for nn in range(len(file_list)):
    for nn in range(1):


        fn2 = path_dir + "//" + file_list[nn]
        count = 0
        cstm_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        #    fn2 = path_dir + "//10Hz_edited.txt"
        data = pd.read_table(fn2, delimiter=r"\s+")
        time = pd.to_numeric(data['Index']) / 10000
        S0 = pd.to_numeric(data['S0(mW)'])
        S1 = pd.to_numeric(data['S1'])
        S2 = pd.to_numeric(data['S2'])
        S3 = pd.to_numeric(data['S3'])
        shot = str(nn)
        ax, fig01 = PS3(shot)

        cm = np.linspace(0, 1, len(S1))  # color map
        cm[-1] = 1.3

        ax.plot(S1, S2, S3, color='c', marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=3)

        S = np.vstack((np.array(S1),np.array(S2),np.array(S3)))
        print("S =", S[:,1:5])
        '''
        S = basistonormal(S)

        S1 = S[0,:]
        S2 = S[1, :]
        S3 = S[2,:]

        ax.plot(S1, S2, S3, color='b', marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=3)
        '''

        # plot STOKES parameters

        fsize = 11
        plt.rc('font', size=fsize)  # controls default text sizes
        plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
        plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
        plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
        plt.rc('legend', fontsize=fsize)  # f-size legend
        plt.rc('axes', titlesize=fsize)  # f-size of the axes title (??)
        plt.rc('figure', titlesize=fsize)  # f-size of the figure title

        fig, ax = plt.subplots(4, figsize=(9/2.54, 7/2.54))
        plt.subplots_adjust(left=0.3, bottom=0.27, right=0.93, top=0.93, wspace=0.2, hspace=0.12)
        ax[0].plot(time, S0, 'k')
        ax[1].plot(time, S1, 'k')
        ax[2].plot(time, S2, 'k')
        ax[3].plot(time, S3, 'k')
        ax[0].set_ylabel('S0')
        ax[1].set_ylabel('S1')
        ax[2].set_ylabel('S2')
        ax[3].set_ylabel('S3')
        ax[3].set_xlabel('time (s)')

        ax[0].set(xlim=(0, .5), ylim=(6.412, 6.44))
        ax[0].set_xticks([])
        ax[1].set(xlim=(0, .5), ylim=(0.444, 0.474))
        ax[1].set_xticks([])
        ax[2].set(xlim=(0, .5), ylim=(-0.724,-0.686))
        ax[2].set_xticks([])
        ax[3].set(xlim=(0, .5), ylim=(-0.571, -0.504))

        fig.align_ylabels()
        fig_name = 'Figure 4(b)' + plt_fmt
        plt.savefig(fig_name, dpi=plt_res)


        fig, ax = plt.subplots(figsize=(7/2.54, 4/2.54))
        plt.subplots_adjust(left=0.17, bottom=0.29, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        Sn = np.ones((len(S0)))
        SS = np.vstack((Sn, S1, S2, S3))
        Out = Sv.from_matrix(SS.T)
        azi_V = Out.parameters.azimuth()
        ellip_V = Out.parameters.ellipticity_angle()
        d_azi_V = azi_V-azi_V[0]
        d_azi_V = d_azi_V - d_azi_V.min()
        d_ellip_V = -(ellip_V-ellip_V[0])
        d_ellip_V = d_ellip_V-d_ellip_V.min()

        alpha = np.sqrt((d_azi_V/cos(ellip_V[0]))**2 + d_ellip_V**2) *180/pi
        ax.plot(time, alpha, 'k', label='alpha')
        ax.plot(time, d_azi_V*180/pi,'r', label='d_phi')
        ax.plot(time, d_ellip_V*180/pi, 'b', label='d_chi')
        #ax.set(xlim=(0, .5), ylim=(0,2))
        ax.set(xlim=(0, .5), ylim=(0,2.5))
        ax.set_xlabel('time (s)')
        ax.set_ylabel('SOP change (deg)')
        ax.legend(loc="upper right")
        fig_name = 'Figure 4(c)' + plt_fmt
        plt.savefig(fig_name, dpi=plt_res)

if (__name__ == "__main__"):
    main()
    plt.show()

