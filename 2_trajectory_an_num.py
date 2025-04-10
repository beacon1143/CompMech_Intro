import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# PHYSICS
v0 = 10
alpha = math.radians(45.0)
g = 9.81

flight_range = v0 * v0 * math.sin(2 * alpha) / g
vx = v0 * math.cos(alpha)
vy = v0 * math.sin(alpha)

# NUMERIC
nsteps = 100
dx = flight_range / nsteps
dt = dx / vx

# PREPROCESSING
x = np.linspace(0, flight_range, nsteps + 1)
y_an = x * math.tan(alpha) - g * x * x / (2.0 * v0 * v0 * math.cos(alpha) ** 2)
y_num = np.zeros(nsteps + 1)

# FIGURE
fig, ax = plt.subplots()
ax.axis('equal')
traject_an = ax.plot(x, y_an, lw=3)[0]
traject_num = ax.plot(x[0], y_num[0], lw=3)[0]

def init_anim():
    traject_an.set_data(x, y_an)
    traject_num.set_data(x[0], y_num[0])
    return (traject_an, traject_num)

def loop_anim(i):
    global vy
    y_num[i + 1] = y_num[i] + vy * dt
    vy = vy - g * dt
    traject_num.set_data(x[:i+2], y_num[:i+2])
    return (traject_an, traject_num)

ani = anim.FuncAnimation(
    fig=fig, func=loop_anim, init_func=init_anim, 
    frames=nsteps, interval=1, repeat=False
)

plt.show()
