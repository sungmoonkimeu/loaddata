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
from py_pol.mueller import create_Mueller


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

path_dir = 'Data_pol3_edited'
file_list = os.listdir(path_dir)


fig, ax = plt.subplots(4, figsize=(6, 5))
Ev = Jones_vector('Output_J')
Sv = create_Stokes('Output_S')
Out = create_Stokes('Output_S2')

LVP = create_Stokes('Vertical linear pol')
LHP = create_Stokes('Horizontal linear pol')
L45P = create_Stokes('+45deg linear pol')
L135P = create_Stokes('-45deg linear pol')
RCP = create_Stokes('Right handed circular pol')
LCP = create_Stokes('Left handed circular pol')


fn2 = path_dir + "//LP0_edited.txt"
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
LHP = Out[0]

fn2 = path_dir + "//LP90_edited.txt"
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
LVP = Out[0]

fn2 = path_dir + "//LP45_edited.txt"
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
L45P = Out[0]

fn2 = path_dir + "//LP135_edited.txt"
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
L135P = Out[0]

fn2 = path_dir + "//LHC_edited.txt"
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
LCP = Out[0]

fn2 = path_dir + "//RHC_edited.txt"
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
RCP = Out[0]


fig2, ax2 = LVP.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='scatter')
cstm_color = ['c', 'm', 'y', 'k', 'r']

draw_stokes_points(fig2[0], LVP, kind='scatter', color_line=cstm_color[0])
draw_stokes_points(fig2[0], LHP, kind='scatter', color_line=cstm_color[1])
draw_stokes_points(fig2[0], L45P, kind='scatter', color_line=cstm_color[2])
draw_stokes_points(fig2[0], L135P, kind='scatter', color_line=cstm_color[3])
draw_stokes_points(fig2[0], LCP, kind='scatter', color_line=cstm_color[4])
draw_stokes_points(fig2[0], RCP, kind='scatter', color_line=cstm_color[0])

M0 = (LHP.parameters.matrix()+LVP.parameters.matrix())/2
M1 = (LHP.parameters.matrix()-LVP.parameters.matrix())/2
M2 = (L45P.parameters.matrix()-L135P.parameters.matrix())/2
M3 = (RCP.parameters.matrix()-LCP.parameters.matrix())/2

M = create_Mueller('Compensation')

M.from_matrix(np.hstack((M0,M1,M2,M3)))

plt.show()
LVP2 = M*LVP
LHP2 = M*LHP
L45P2 = M*L45P
L135P2 = M*L135P
LCP2 = M*LCP
RCP2 = M*RCP

cstm_color = ['c', 'm', 'y', 'k', 'r']

draw_stokes_points(fig2[0], LVP2, kind='scatter', color_line=cstm_color[0])
draw_stokes_points(fig2[0], LHP2, kind='scatter', color_line=cstm_color[1])
draw_stokes_points(fig2[0], L45P2, kind='scatter', color_line=cstm_color[2])
draw_stokes_points(fig2[0], L135P2, kind='scatter', color_line=cstm_color[3])
draw_stokes_points(fig2[0], LCP2, kind='scatter', color_line=cstm_color[4])
draw_stokes_points(fig2[0], RCP2, kind='scatter', color_line=cstm_color[0])

