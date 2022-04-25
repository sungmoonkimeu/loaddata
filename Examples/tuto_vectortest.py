import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(0, np.pi, 15)  # azimuth
v = np.linspace(0, 2 * np.pi, 11)  # elevation
sprad = 1
x = sprad * np.outer(np.sin(u), np.cos(v))
y = sprad * np.outer(np.sin(u), np.sin(v))
z = sprad * np.outer(np.cos(u), np.ones(np.size(v)))

fig, ax = plt.subplots(len(v),figsize=(6,15))
fig2, ax2 = plt.subplots(len(v),figsize=(6,15))
fig3, ax3 = plt.subplots(len(v),figsize=(6,15))
for nn in range(len(v)):

    ax[nn].plot(x[nn])
    ax[nn].set(ylim=(-1,1))
    ax2[nn].plot(y[nn])
    ax2[nn].set(ylim=(-1, 1))
    ax3[nn].plot(z[nn])
    ax3[nn].set(ylim=(-1, 1))
plt.show()