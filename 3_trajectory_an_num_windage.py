import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS
v0 = 100
alpha = math.radians(45.0)
g = 9.81
windage = 0.01

flight_range = v0 * v0 * math.sin(2 * alpha) / g
vx = v0 * math.cos(alpha)
vy = v0 * math.sin(alpha)

# NUMERIC
nsteps_an = 400
nsteps_num = 40000
dx = flight_range / nsteps_an
dt = dx / vx

# PREPROCESSING
x_an = np.linspace(0, flight_range, nsteps_an + 1)
y_an = x_an * math.tan(alpha) - g * x_an * x_an / (2.0 * v0 * v0 * math.cos(alpha) ** 2)
x_num = np.zeros(nsteps_num + 1)
y_num = np.zeros(nsteps_num + 1)

# ACTION
fig, ax = plt.subplots()
ax.axis('equal')
traject_an = ax.plot(x_an, y_an, lw=3)[0]
traject_num = ax.plot(x_num[0], y_num[0], lw=3)[0]

def init_anim():
    traject_an.set_data(x_an, y_an)
    traject_num.set_data(x_num[0], y_num[0])
    return (traject_an, traject_num)

def loop_anim(i):
    global vx, vy
    x_num[i + 1] = x_num[i] + vx * dt
    vx = vx * (1.0 - windage * dt)
    y_num[i + 1] = y_num[i] + vy * dt
    vy = vy - (g + windage * vy) * dt
    if y_num[i + 1] <= 0:
        return (traject_an, traject_num)
    traject_num.set_data(x_num[:i+2], y_num[:i+2])
    return (traject_an, traject_num)

ani = animation.FuncAnimation(
    fig=fig, func=loop_anim, init_func=init_anim, 
    frames=nsteps_num, interval=10, repeat=False
)

plt.show()
