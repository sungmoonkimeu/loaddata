#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:52:20 2021

@author: prasadarajudandu

Beat length measurement in a spun fibre using POTDR method
"""
import numpy as np
from numpy import cos,pi,mat,concatenate,ones,zeros,arctan,arcsin,tan,sin,arange,sqrt,append
from numpy.linalg import norm
import matplotlib.pyplot as plt
LB0    = 0.32                #beat lenth
SP    = 0.08               #spun period [m / 1 turn]
L     = 0.8 #0.85#0.45      #length of the fiber
A_P   = 45*(pi/180)#pi/2    #polarization angle
STR   = 2*pi/SP             #spin rate  [rad / m]

#V_in  = mat([[0.707], [-0.707*1j]])#mat([[cos(A_P)],[sin(A_P)]])        #Normalized input Jones vector
th    = 45 * pi/180
#V_in  = mat([[0.707], [0.707]])        #Normalized input Jones vector

V_in  = mat([[cos(th)], [sin(th)*np.exp(1j*30*pi/180)]])        #Normalized input Jones vector
M_P   = mat([[(cos(A_P))**2, (sin(A_P)*cos(A_P))],
            [(sin(A_P)*cos(A_P)), (sin(A_P))**2]])   #polarizer matrix

print(V_in)
dz = 0.0001  #element length

def eigen_expm(A):
            """

            Parameters
            ----------
            A : 2 x 2 diagonalizable matrix
                DESCRIPTION.

            Returns
            -------
            expm(A): exponential of the matrix A.

            """
            vals,vects = np.linalg.eig(A)
            return np.einsum('...ik, ...k, ...kj -> ...ij',
                             vects,np.exp(vals),np.linalg.inv(vects))

V_L      = append(arange(dz,L,dz),L) #length vector
V_dz     = dz*ones(len(V_L)) #vector of element lengths
BW = 20
#PB  = zeros([len(delta_wl), len(V_L)])
maximum_ER = 10  # maximum extinction ratio[db] noise level is -8 dBm
applied_twist = arange(2, 18, 1)
PB  = zeros([len(applied_twist), len(V_L)])
PB2 = zeros([len(applied_twist), len(V_L)])
twist0 = L/SP
avg_length = 10  # 50 mm average
avg_number = int(L/dz*avg_length/1000)  # 10 mm average number
for j, twist in enumerate(applied_twist):
    LB = LB0
    SP2 = L/twist
    STR2 = -2*pi/SP2
    #V_in = mat([[-0.707], [0.707 * np.exp(-LB * 0.03 * 1j)]])  # Normalized input Jones vector
    Delta = 2*pi/LB                 #linear birefringnece
    V_delta = Delta*ones(len(V_L))  # lin. biref vector

    g    = 0.073 # photoelastic coefficeint
    sign = -1    # sign of twist, +1/-1 are for twisting in the direction and against twist
    tau  = STR2 #2*pi*9.528# # twist rate
    af   = g*tau # twist induced circular birefringence
    #print(STR+ tau)
    q0=0 #initial azimuth of the spun fiber
    t_f    = (STR+tau)*V_dz        # birefringence axis rotation azimuth of each element due to spinning(forward direction)
    t_f[0] = 0 #for the first section the azimuth doesn't change, it remains same as that at the entrance.
    t_s_f  = q0+np.cumsum(t_f)
    t_b    = (STR+tau)*V_dz # birefringence axis rotation azimuth of each element due to spinning (backward direction)
    t_s_b  = q0+np.cumsum(t_b)
    #-----------------------------------------------------------------------------
    #The following parameters are defined as per Laming (1989) paper
    dlt_f= sqrt((V_delta/2)**2+((STR+tau-af))**2)
    dlt_b= sqrt((V_delta/2)**2+(-(STR+tau-af))**2)

    R_z_f=2*arcsin((V_delta/2)*(sin(dlt_f*V_dz)/dlt_f))
    R_z_b=2*arcsin((V_delta/2)*(sin(dlt_b*V_dz)/dlt_b))

    omega_z_f=(STR+tau)*V_dz+arctan((-(STR+tau-af))*(tan(dlt_f*V_dz)/dlt_f))
    omega_z_b=-(STR+tau)*V_dz+arctan(((STR+tau-af))*(tan(dlt_b*V_dz)/dlt_b))

    phi_z_f=((STR+tau)*V_dz-omega_z_f)/2+t_s_f
    phi_z_b=(-(STR+tau)*V_dz-omega_z_b)/2+t_s_b

    theta_R_laming_f_all=np.reshape((R_z_f/2),(len(V_dz),1,1))*np.array([[1j*
     cos(2*phi_z_f),1j*sin(2*phi_z_f)],[1j*sin(2*phi_z_f),-1j*cos(2*phi_z_f)]]).transpose() # retardation N matrix (forward direction)

    theta_omg_laming_f_all = np.einsum('...i,jk->ijk',omega_z_f,np.mat([[0,-1],[1,0]])) #rotation N matrix (forward direction)

    N_i_f_all              = theta_R_laming_f_all+theta_omg_laming_f_all #Jones N matrix in the forward direction

    theta_R_laming_b_all=np.reshape((R_z_b/2),(len(V_dz),1,1))*np.array([[1j*
    cos(2*phi_z_b),1j*sin(2*phi_z_b)],[1j*sin(2*phi_z_b),-1j*cos(2*phi_z_b)]]).transpose() # retardation N matrix (backward direction)

    theta_omg_laming_b_all = np.einsum('...i,jk->ijk',omega_z_b,np.mat([[0,-1],[1,0]])) #rotation N matrix(backward direction)

    N_i_b_all              = theta_R_laming_b_all+theta_omg_laming_b_all #Jones N matrix in the backward direction

    M_i_f = eigen_expm(N_i_f_all)
    M_i_b = eigen_expm(N_i_b_all)

    M_f = mat([[1, 0], [0, 1]])
    M_b = mat([[1, 0], [0, 1]])
    #PB  = zeros(len(V_L))
    PB_avg = zeros(len(V_L))
    for i in range(len(V_L)):

        M_f        = M_i_f[i]*M_f  # forward jones matrix
        M_b        = M_b*M_i_b[i]  # backward jones matrix

        Vout       = M_P*M_b*M_f*V_in # o/p SOP
        PB[j,i]      = ((norm(Vout))**2)
        '''
        if PB[j,i] < 10 ** (-maximum_ER/10):
            PB[j, i] = 10 ** (-maximum_ER/10)
        '''
        PB[j, i] += 10 ** (-maximum_ER / 10)

    for i in range(len(V_L)):
        if i == 0:
            PB2[j, i] = PB[j, i]
        elif i < avg_number:
            PB2[j, i] = np.average(PB[j, 0:i])
        else:
            PB2[j, i] = np.average(PB[j, i-avg_number:i])


    #plt.plot(V_L, 10 * np.log10(PB[j]), label='80% compensation')  # 10*np.log10(PB)

for i in range(len(V_L)):
    PB_avg[i] = np.mean(PB[:, i])

#plt.plot(V_L, PB, label='80% compensation')#10*np.log10(PB)

for j in range(len(applied_twist)):
    plt.plot(V_L, 10*np.log10(PB2[j]),
             label=str(applied_twist[j]) + "turns, " +  str(int((applied_twist[j] - twist0)/twist0*100)) + '% compensation')#10*np.log10(PB)
    plt.legend()


## FFT

fig, ax = plt.subplots(len(applied_twist), figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))
#fig3, ax3 = plt.subplots(len(applied_twist), figsize=(6, 5))

maxdatay = np.array([])
maxdatax = np.array([])
#PB2 = zeros([len(applied_twist), len(V_L)])

for nn, fn in enumerate(PB2):
    data = PB2[nn]
    length = V_L

    xmin = 3.8
    xmax = 4.5
    ymin = -134
    ymax = -116
    data2 = 10**(data/10)
    data2 = data2 - data2.mean()
    fs = 1 / (length[2]-length[1])

    fdata = np.fft.fft(data2)/len(data2)
    xdata = np.linspace(0, fs, len(fdata))
    ax[nn].plot(xdata, 2*abs(fdata), lw='1',
                label=str(applied_twist[nn]) + "turns, ")
    ax[nn].set(xlim=(0, 30), ylim=(0, 0.2))
    ax[nn].legend(loc="upper right")

    maxdatax = np.append(maxdatax, nn)
    maxdatay = np.append(maxdatay, 2/xdata[np.argmax(abs(fdata[0:100]))])
    print(np.argmax(abs(fdata)))

    if nn == 5:
        figx, axx = plt.subplots()
        axx.plot(xdata, 2 * abs(fdata))
        print(xdata[0:20])
        print(2 * abs(fdata[0:20]))

ax[-1].set_xlabel('Frequency (1/m)')
ax[int(len(ax)/2)].set_ylabel('FFT')
ax2.plot(maxdatax, maxdatay)




plt.show()
