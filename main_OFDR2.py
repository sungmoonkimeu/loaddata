"""Analysing datafiles from OFDR device.
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
import os

# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Computer Modern'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['font.sans-serif']='Comic Sans MS'

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


def switch_osfolder(path1, path2):
    try:
        if os.path.exists(path1):
            os.chdir(path1)
        else:
            os.chdir(path2)
    except OSError:
        print('Error: Changing OS directory')


# Switching OS folder
path2 = 'C:/Users/Iter/PycharmProjects/loaddata'
path1 = 'C:/Users/SMK/PycharmProjects/loaddata'

switch_osfolder(path1, path2)

foldername = 'Data_1308//spunfiber//1st'
foldername = 'Data_1308//withouthspun'


path_dir = os.getcwd() + '//Data_Twsiting_(OFDR)//' + foldername + '_edited'
file_list = os.listdir(path_dir)
'''
# 1st step see the data from file
for nn, _fn in enumerate(file_list):
    fn = path_dir + "//" + _fn
    data = pd.read_table(fn)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(time, signal, lw='1', label=file_list[nn])
    ax.legend(loc="upper right")
    #ax.set(xlim=(35, 46), ylim=(-134, -116))
    ax.set(xlim=(109, 120), ylim=(-140, -125))
    if nn > 3:
        break
plt.show()
'''


# 2nd step see the named file
a = arange(1, 8, 1)
count = 0
fig, ax = plt.subplots(len(a), figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))
for nn in a:

    #fn2 = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
    fn2 = path_dir + "//" + str(nn) + "t_ac_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10
    ax[count].plot(length, signal, lw='1', label=str(nn)+"turn")
    ax[count].legend(loc="upper left")
    # ax[count].set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax[count].set(xlim=(10.9, 12.0), ylim=(-140, -125))

    ax2.plot(length, signal, lw='1', label=str(nn)+"turn")
    ax2.plot(length, signal, lw='1', label=str(nn) + "turn")
    ax2.legend(loc="upper left")
    # ax2.set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax2.set(xlim=(10.9, 12.0), ylim=(-140, -125))

    count = count+1

ax[-1].set_xlabel('Length (m)')
ax[int(len(ax)/2)].set_ylabel('Power (dB/mm)')
ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Power (dB/mm)')

'''
# 3rd step see the named file
a = arange(210, 330, 30)
count = 0
fig, ax = plt.subplots(len(a), figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))

for nn in a:

    fn2 = path_dir + "//" + "9t_" + str(nn) + "deg_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10
    ax[count].plot(length, signal, lw='1', label=str(nn)+"deg")
    ax[count].legend(loc="upper left")
    ax[count].set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax2.plot(length, signal, lw='1', label=str(nn) + "turn")
    ax2.legend(loc="upper left")
    ax2.set(xlim=(3.5, 4.6), ylim=(-134, -116))
    count = count+1

ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Power (dB/mm)')
'''
'''
a = arange(30, 361, 30)
count = 0
#fig, ax = plt.subplots(len(a), figsize=(6, 5), left=0.093, bottom = 0.07, right = 0.96, top = 0.967, wspace = 0.2, hspace = 0 )
fig, ax = plt.subplots(len(a), figsize=(6, 5))
plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

for nn in a:

    fn2 = path_dir + "//" + "9turns_" + str(nn) + "deg_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = 10**(data['Amplitude (dB)']/10)
    ax[count].plot(time/10, signal, lw='1', label=str(nn)+"deg_2nd")
    ax[count].legend(loc="upper left")
    #ax[count].set(xlim=(11.8, 13.7), ylim=(-135, -120))
    ax[count].set(xlim=(11.8, 13.7), ylim=(10**(-13.7), 10**(-12.5)))
    count = count+1
'''

'''
a = arange(120, 150, 30)
fig, ax = plt.subplots(figsize=(6, 3))
plt.subplots_adjust(left=0.145, bottom= 0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

for nn in a:

    fn2 = path_dir + "//" + "9turns_" + str(nn) + "deg_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = (10**(data['Amplitude (dB)']/10))
    #ax.plot(time/10, signal, lw='1', label=str(nn)+"deg_2nd")
    ax.plot(time / 10, signal, lw='1', label="9turns + " + str(nn) + "deg (" + str(round(9 + nn/360, 2)) + "turns)")
    ax.legend(loc="upper left")
    #ax.set(xlim=(11.8, 13.7), ylim=(-135, -120))
    ax.set(xlim=(11.8, 13.7), ylim=(10**(-13.7), 10**(-12.5)))
    count = count+1
'''
'''
fig, ax = plt.subplots(figsize=(6, 3))
plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)
ax.set(xlim=(11.8, 13.7), ylim=(10 ** (-13.7), 10 ** (-12.5)))

#tmpdata = -0.70e-13*cos(2*pi*0.4*(time+10))+1.2e-13
# for 120 deg matching
tmpdata = -0.72e-13*cos(2*pi*0.445*(time+10))+1.2e-13
ax.plot((time+134.7) / 10, tmpdata, lw='2', label="fitting for " + str(round(9 + 120/360, 2)) + "turns")
# for 150 deg matching
#tmpdata = -0.73e-13*cos(2*pi*0.43*(time+10))+1.18e-13
#ax.plot((time+134.7) / 10, tmpdata, lw='2', label="fitting for " + str(round(9 + 150/360, 2)) + "turns")
# for 180 deg matching3
#tmpdata = -0.82e-13*cos(2*pi*0.418*(time+10))+1.24e-13
#ax.plot((time+134.7) / 10, tmpdata, lw='2', label="fitting for " + str(round(9 + 180/360, 2)) + "turns")
# for 240 deg matching
tmpdata = -0.92e-13*cos(2*pi*0.395*(time+10))+1.4e-13
ax.plot((time+134.7) / 10, tmpdata, lw='2', label="fitting for " + str(round(9 + 240/360, 2)) + "turns")
# for 360 deg matching
tmpdata = -1.01e-13*cos(2*pi*0.36*(time+10))+1.5e-13
ax.plot((time+134.7) / 10, tmpdata, lw='2', label="fitting for " + str(round(9 + 360/360, 2)) + "turns")

ax.legend(loc="upper left")

#plt.rc('text', usetex=True)
ax.set_xlabel('Length (m)')
ax.set_ylabel('Power (mW/mm)')
'''
'''
path_dir = 'Data2_edited'
file_list = os.listdir(path_dir)

a = arange(30, 360, 30)
count = 0
for nn in a:

    fn2 = path_dir + "//" + "9_" + str(nn) + "deg_hibi_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    ax[count].plot(time/10, signal, lw='1', label=str(nn)+"deg_1st")
    ax[count].legend(loc="upper left")
    ax[count].set(xlim=(11.8, 13.7), ylim=(-135, -120))
    count = count+1

plt.xlabel("length (m)")
fig.text(0.03, 0.5, 'Amplitue (dB/mm)', ha='center', va='center', rotation='vertical')

# polarization change
path_dir = 'Data2_edited'
file_list = os.listdir(path_dir)
a = arange(1, 10, 1)
fig, ax = plt.subplots(len(a), figsize=(6, 5))

count = 0
for nn in a:

    fn2 = path_dir + "//" + "9turns" + str(nn) + "_hibi_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    ax[count].plot(time/10, signal, lw='1', label='pol. state '+str(nn))
    ax[count].legend(loc="upper left")
    ax[count].set(xlim=(11.8, 13.7), ylim=(-137, -120))
    count = count+1

plt.xlabel("length (m)")
fig.text(0.03, 0.5, 'Amplitue (dB/mm)', ha='center', va='center', rotation='vertical')

plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

# large scale

path_dir = 'Data2_edited'
file_list = os.listdir(path_dir)
a = arange(9, 20, 1)
fig, ax = plt.subplots(len(a), figsize=(6, 5))

count = 0
for nn in a:

    fn2 = path_dir + "//" + str(nn) + "_hibi_Upper_edited.txt"

    if nn == 9:
        fn2 = path_dir + "//" + str(nn) + "turns1_hibi_Upper_edited.txt"

    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    ax[count].plot(time/10, signal, lw='1', label=str(nn)+"turns")
    ax[count].legend(loc="upper left")
    ax[count].set(xlim=(11.8, 13.7), ylim=(-137, -120))
    count = count+1

plt.xlabel("length (m)")
fig.text(0.03, 0.5, 'Amplitue (dB/mm)', ha='center', va='center', rotation='vertical')

plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

# Opposite direction


path_dir = 'Data2_edited'
file_list = os.listdir(path_dir)
a = arange(12, 7, -1)
fig, ax = plt.subplots(len(a), figsize=(6, 5))

count = 0
for nn in a:

    fn2 = path_dir + "//" + "opp_" + str(nn) + "turns_hibi_Upper_edited.txt"

    if nn == 7:
        fn2 = path_dir + "//after twisting_0turn_hibi_Upper_edited.txt"

    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    if nn == 7:
        ax[count].plot(time/10, signal, lw='1', label="0 turns")
    else:
        ax[count].plot(time / 10, signal, lw='1', label="- " + str(nn) + "turns")

    ax[count].legend(loc="upper left")
    ax[count].set(xlim=(11.8, 13.7), ylim=(-137, -120))
    count = count+1

plt.xlabel("length (m)")
fig.text(0.03, 0.5, 'Amplitue (dB/mm)', ha='center', va='center', rotation='vertical')

plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

# inspection
path_dir = 'Data3_edited'
file_list = os.listdir(path_dir)

a = 180
# fig, ax = plt.subplots(len(a), figsize=(6, 5), left=0.093, bottom = 0.07, right = 0.96, top = 0.967, wspace = 0.2, hspace = 0 )
fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(left=0.145, bottom=0.07, right=0.96, top=0.967, wspace=0.2, hspace=0)

fn2 = path_dir + "//" + "9turns_180deg_Upper_edited.txt"
data = pd.read_table(fn2)
time = data['Time (ns)']
signal = data['Amplitude (dB)']
ax.plot(time/10, signal, lw='1', label=str(a)+"deg_2nd")
ax.legend(loc="upper left")
ax.set(xlim=(11.8, 13.7), ylim=(-135, -120))
count = count+1

'''


# 2nd step see the named file
a = arange(1, 8, 1)
b = arange(2, 0, -1)
count = 0
fig, ax = plt.subplots((len(a) + len(b)+1), figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))

for nn in b:
    fn2 = path_dir + "//" + str(nn) + "t_c_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10
    ax[count].plot(length, signal, lw='1', label=str(nn)+"turn")
    ax[count].legend(loc="upper left")
    # ax[count].set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax[count].set(xlim=(10.9, 12.0), ylim=(-140, -125))
    ax2.plot(length, signal, lw='1', label=str(-nn) + "turn")

    ax2.legend(loc="upper left")
    # ax2.set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax2.set(xlim=(10.9, 12.0), ylim=(-140, -125))

    count = count+1

fn2 = path_dir + "//lin_biref_base_mes_Upper_edited.txt"
data = pd.read_table(fn2)
time = data['Time (ns)']
signal = data['Amplitude (dB)']
length = time / 10
ax[count].plot(length, signal, lw='1', label="0 turn")
ax[count].set(xlim=(10.9, 12.0), ylim=(-140, -125))
ax[count].legend(loc="upper left")

ax2.plot(length, signal, lw='1', label="0 turn")
ax2.set(xlim=(10.9, 12.0), ylim=(-140, -125))

count = count+1

for nn in a:
    # fn2 = path_dir + "//" + str(nn) + "t_Upper_edited.txt"
    fn2 = path_dir + "//" + str(nn) + "t_ac_Upper_edited.txt"
    data = pd.read_table(fn2)
    time = data['Time (ns)']
    signal = data['Amplitude (dB)']
    length = time / 10
    ax[count].plot(length, signal, lw='1', label=str(nn) + "turn")
    ax[count].legend(loc="upper left")
    # ax[count].set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax[count].set(xlim=(10.9, 12.0), ylim=(-140, -125))

    ax2.plot(length, signal, lw='1', label=str(nn) + "turn")
    ax2.legend(loc="upper left")
    # ax2.set(xlim=(3.5, 4.6), ylim=(-134, -116))
    ax2.set(xlim=(10.9, 12.0), ylim=(-140, -125))

    count = count + 1

ax[-1].set_xlabel('Length (m)')
ax[int(len(ax)/2)].set_ylabel('Power (dB/mm)')
ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Power (dB/mm)')

plt.show()
