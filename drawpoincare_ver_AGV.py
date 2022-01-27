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


plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", message="This figure includes Axes")
warnings.filterwarnings("ignore", message="This figure was using")
warnings.filterwarnings("ignore", message="Calling figure.constrained_layout")

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
    ax.view_init(elev=-158, azim=126)
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
    plt.close("all")
    plt_fmt, plt_res = '.png', 330  # 330 is max in Word'16
    plt.rcParams["axes.titlepad"] = 20  # offset for the fig title
    #  plt.rcParams['figure.constrained_layout.use'] = True  # fit legends in fig window
    fsize = 18
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
    plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('legend', fontsize=fsize)  # f-size legend
    plt.rc('axes', titlesize=6)  # f-size of the axes title (??)
    plt.rc('figure', titlesize=6)  # f-size of the figure title
    #

    m01_shots = [90000, 99700]
    #focs_dir = 'Shots\\'
    #focs_out = 'Shots-PS\\'
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

        #ax.scatter3D(S1, S2, S3, zdir='z', marker='o', s=4, c=cm, zorder=555,
        #             alpha=1, label='F1', cmap="cool")

        ax.plot(S1, S2, S3, marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=3)

        #ax.scatter3D(D1, D2, D3, zdir='z', marker='+', s=6, c=cm,
        #             alpha=0.6, label='F2', cmap="cool")
        #ax.legend()

if (__name__ == "__main__"):
    main()
    plt.show()

