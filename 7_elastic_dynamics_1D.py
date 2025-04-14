import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# PHYSICS
lx = 10.0
K = 1.0
G = 0.5
rho = 1.0

# NUMERICS
nx = 200
nsteps = 400
cfl = 1.0
dx = lx / (nx - 1)
dt = cfl * dx / np.sqrt((K + 4.0 * G / 3.0) / rho)

# PREPROCESSING
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)

# INITIAL CONDITIONS
p0 = 1.0
p = p0 * np.exp(-x * x)
tau = np.zeros(nx)
v = np.zeros(nx + 1)

# ACTION
fig, ax = plt.subplots()
cur_line = ax.plot(x, p, lw=3)[0]

def loop_anim(i):
    global p, tau
    fig.suptitle(str(i + 1))
    dpdt = -np.diff(v) / dx * K
    p = p + dpdt * dt
    dtaudt = np.diff(v) / dx * 4.0 * G / 3.0
    tau = tau + dtaudt * dt
    dvdt = np.diff(-p + tau) / dx / rho
    v[1:-1] = v[1:-1] + dvdt * dt
    cur_line.set_data(x, p)
    #ax.set_ylim(p.min(), p.max())
    return (cur_line)

ani = anim.FuncAnimation(fig=fig, func=loop_anim, frames=nsteps, interval=1, repeat=False)
plt.show()