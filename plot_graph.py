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
from scipy.signal import welch

import matplotlib.transforms
import pandas as pd
import os


g = 0.14  # stress-optic coefficient (0.14~ 0.16)
#tau = 10  # twist rate (rad/m)
#tl = arange(0.0001, 1000, 0.000001)
tl = np.logspace(-2., 5., num=10000)

tau = 2*pi/tl
Lb = 1
beta_lin = 2*pi/Lb   # beatlength_linear
xi = 2*pi / 0.08

Lb = 2*pi/sqrt(beta_lin**2 + (g*tau - 2*tau)**2)
Lb2 = 2*pi/sqrt(beta_lin**2 + (2*xi + g*tau - 2*tau)**2)
fig, ax = plt.subplots(figsize=(6, 5))
#ax.set(xlim=(0, 100), ylim=(0, 30))

ax.plot(tl, Lb)
ax.plot(tl, Lb2)
ax.set_xlabel('twist length (m)')
ax.set_ylabel('beatlength (m)')

fig, ax = plt.subplots(figsize=(6, 5))
#ax.set(xlim=(0, 100), ylim=(0, 30))

ax.plot(tau, Lb)
ax.plot(tau, Lb2)
ax.set_xlabel('twist rate (rad/m)')
ax.set_ylabel('beatlength (m)')

plt.show()