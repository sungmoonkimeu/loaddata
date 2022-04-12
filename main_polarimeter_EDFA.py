"""Analysing datafiles from Polarimeter device
"""

import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt

import pandas as pd
from matplotlib.lines import Line2D

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import basis_calibration
import os

foldername = '//EDFA_stability'
path_dir = os.getcwd() + foldername + '_edited'

file_list = os.listdir(path_dir)

Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

# Draw empty Poincare sphere
fig2, ax2 = Sv.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line')

diff_azi_V = np.ones(len(file_list))
diff_ellip_V = np.ones(len(file_list))

max_diff_S = np.zeros([2, 4])
mean_S = np.zeros([2, 4])

fig, ax1 = plt.subplots(4, figsize=(6, 5))

for nn in range(len(file_list)):

    fn2 = path_dir + "//" + file_list[nn]
    print("filename = ", file_list[nn])
    count = 0
    cstm_color = ['y', 'b', 'r', 'k', 'g']

    data = pd.read_table(fn2, delimiter=r"\s+")

    S0x = pd.to_numeric(data['S0(mW)'])
    S1 = pd.to_numeric(data['S1'])
    S2 = pd.to_numeric(data['S2'])
    S3 = pd.to_numeric(data['S3'])
    time = np.arange(0, len(S0x), 1) / 1800

    ndata = 720  # for one hour
    #ndata = len(S0) - 720 # for all data

    Sn = np.ones((len(S0x)))
    #SS = np.vstack((Sn[720::2], S1[720::2], S2[720::2], S3[720::2]))
    #SS = np.vstack((Sn[720:720+ndata:2], S1[720:720+ndata:2], S2[720:720+ndata:2], S3[720:720+ndata:2]))
    SS = np.vstack((Sn, S1, S2, S3))
    Out = Sv.from_matrix(SS.T)

    #draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 4])
    '''
    Out = basis_calibration.calib_basis2(Out)
    S0 = Out.parameters.matrix()[0]
    S1 = Out.parameters.matrix()[1]
    S2 = Out.parameters.matrix()[2]
    S3 = Out.parameters.matrix()[3]
    '''
    draw_stokes_points(fig2[0], Out, kind='line', color_line=cstm_color[nn % 5])

    azi_V = Out.parameters.azimuth()
    ellip_V = Out.parameters.ellipticity_angle()

    diff_azi_V[nn] = azi_V.max() - azi_V.min()
    diff_ellip_V[nn] = ellip_V.max() - ellip_V.min()
    print('maximum SOP change=', sqrt(diff_azi_V[nn]**2 + diff_ellip_V[nn]**2)*180/pi, "deg")


    ax1[0].plot(time, S0x)
    ax1[0].set_ylabel("S" + str(0))
    # ax[0].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax1[1].plot(time, S1)
    ax1[1].set_ylabel("S" + str(1))
    # ax[1].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax1[2].plot(time, S2)
    ax1[2].set_ylabel("S" + str(2))
    # ax[2].set(xlim=(0, 0.5), ylim=(-1, 1))
    ax1[3].plot(time, S3)
    ax1[3].set_ylabel("S" + str(3))
    # ax[3].set(xlim=(0, 0.5), ylim=(-1, 1))

    ax1[3].set_xlabel("Time (h)")
    fig.align_ylabels()

custom_lines = [Line2D([0], [0], color=cstm_color[0], lw=4)]
fig2[0].legend(custom_lines, ['EDFA'], loc='right')

plt.show()


