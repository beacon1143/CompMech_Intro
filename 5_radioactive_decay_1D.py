import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# PHYSICS
lx = 10.0
c = -1.0

# NUMERICS
dt = 0.01
nsteps = 400
nx = 200

# PREPROCESSING
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)

# INITIAL CONDITIONS
p0 = 1.0
p_init = p0 * np.exp(-x * x)
p = p_init

# ACTION
fig, ax = plt.subplots()
init_line = ax.plot(x, p_init, lw=3)[0]
cur_line = ax.plot(x, p, lw=3)[0]

def loop_anim(i):
    global p
    fig.suptitle(str(i + 1))
    p = p * (1.0 + c * dt)
    cur_line.set_data(x, p)
    return (init_line, cur_line)

ani = anim.FuncAnimation(fig=fig, func=loop_anim, frames=nsteps, interval=1, repeat=False)

plt.show()