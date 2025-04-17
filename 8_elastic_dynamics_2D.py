import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# PHYSICS
lx = 10.0
ly = 10.0
K = 1.0
G = 0.5
rho = 1.0

# NUMERICS
nx = 200
ny = 200
nsteps = 1000
cfl = 0.5
dx = lx / (nx - 1)
dy = ly / (ny - 1)
dt = cfl * min(dx, dy) / np.sqrt((K + 4.0 * G / 3.0) / rho)

# PREPROCESSING
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
y = np.linspace(-0.5 * ly, 0.5 * ly, ny)
x, y = np.meshgrid(x, y, indexing='ij')    # 1D arrays x and y became 2D

# INITIAL CONDITIONS
p0 = 1.0
p = p0 * np.exp(-x * x - y * y)
tauxx = np.zeros((nx, ny))
tauyy = np.zeros((nx, ny))
tauxy = np.zeros((nx - 1, ny - 1))
vx = np.zeros((nx + 1, ny))
vy = np.zeros((nx, ny + 1))

# ACTION
fig, ax = plt.subplots()
cur_plot = ax.pcolormesh(x, y, p)
fig.colorbar(cur_plot)
ax.axis('scaled')

def loop_anim(i):
    global p, tauxx, tauyy, tauxy
    fig.suptitle(str(i + 1))
    div_v = np.diff(vx, 1, 0) / dx + np.diff(vy, 1, 1) / dy
    p[:] = p - div_v * K * dt
    dtauxxdt = (np.diff(vx, 1, 0) / dx - div_v / 3.0) * 2.0 * G
    tauxx[:] = tauxx + dtauxxdt * dt
    dtauyydt = (np.diff(vy, 1, 1) / dy - div_v / 3.0) * 2.0 * G
    tauyy[:] = tauyy + dtauyydt * dt
    dtauxydt = (np.diff(vx[1:-1, :], 1, 1) / dy + np.diff(vy[:, 1:-1], 1, 0) / dx) * G
    tauxy[:] = tauxy + dtauxydt * dt
    dvxdt = (np.diff(-p[:, 1:-1] + tauxx[:, 1:-1], 1, 0) / dx + np.diff(tauxy, 1, 1) / dy) / rho
    vx[1:-1, 1:-1] = vx[1:-1, 1:-1] + dvxdt * dt
    dvydt = (np.diff(-p[1:-1, :] + tauyy[1:-1, :], 1, 1) / dy + np.diff(tauxy, 1, 0) / dx ) / rho
    vy[1:-1, 1:-1] = vy[1:-1, 1:-1] + dvydt * dt
    cur_plot.set_array(p)
    cur_plot.set_clim([p.min(), p.max()])
    return (cur_plot)

ani = anim.FuncAnimation(fig=fig, func=loop_anim, frames=nsteps, interval=1, repeat=False)

plt.show()