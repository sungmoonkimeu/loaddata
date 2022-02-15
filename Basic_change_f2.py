import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt, exp
from numpy.linalg import norm, eig
from py_pol.jones_matrix import Jones_matrix
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.mueller import Mueller
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.pyplot as plt


E = Jones_vector('Input')
S = create_Stokes('Output')
S2 = create_Stokes('cal')
J1 = Jones_matrix('Random element')
J2 = Jones_matrix('Random element')
M = Mueller('cal')

azi = np.arange(0, pi/2, 0.1)
# ell = np.arange(0, pi/6, 0.1)
E.general_azimuth_ellipticity(azimuth=azi, ellipticity=pi/12)
S.from_Jones(E)
fig, ax = S.draw_poincare(kind='line', color_line='k')

phi0 = -pi/12
Mp = np.array([[exp(1j*phi0/2), 0], [0, exp(-1j*phi0/2)]])
phi1 = pi/12
Mr = np.array([[cos(phi1),-sin(phi1)], [sin(phi1), cos(phi1)]])

J1.from_matrix(Mr)
J2.from_matrix(Mp)
Out = J2*J1*E
S.from_Jones(Out)
# S.from_Jones(Out).draw_poincare()

draw_stokes_points(fig[0], S, kind='line', color_line='r')

a = S.parameters.matrix()[1:]           # convert 4x1 Stokes vectors to 3x1 cartesian vectors
c = a[...,0]

print("new c", c)
fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1,)

z = [0,0,1]
y = [0,1,0]
x = [1,0,0]

th_x = np.arccos(np.dot(x,[c[0],c[1],0]/np.linalg.norm([c[0],c[1],0])))
th_y = np.arccos(np.dot(y,[c[0],c[1],0]/np.linalg.norm([c[0],c[1],0])))
th_z = np.arccos(np.dot(z,c))

print("th_x=", th_x*180/pi, "th_y=", th_y*180/pi, "th_z=", th_z*180/pi)

Rx = np.array([[cos(th_x), -sin(th_x), 0], [sin(th_x), cos(th_x), 0], [0, 0, 1]])
Ry = np.array([[1, 0, 0], [0, cos(th_y), -sin(th_y)], [0, sin(th_y), cos(th_y)]])
Rz = np.array([[cos(th_z), 0, sin(th_z)], [0,1,0], [-sin(th_z), 0, cos(th_z)]])


th = -th_y
if th_x > pi:
    th = th_y
Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1,)


th = 0
R45 = np.array([[cos(th), 0, sin(th)], [0,1,0], [-sin(th), 0, cos(th)]])     # S2, + 기준 rotation

th = pi/2-th_z
Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])   # S1, H 기준 rotation

TT = R45.T@Rh.T@Rr.T@a
zT = ones(np.shape(TT)[1])

Sp = np.vstack((zT,TT))
S.from_matrix(Sp)

draw_stokes_points(fig[0], S, kind='line', color_line='b')

plt.show()
