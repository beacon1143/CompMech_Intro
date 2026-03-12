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
nsteps_an = 40
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

def derivatives(x, y, vx, vy):
    dxdt = vx
    dydt = vy
    dvxdt = -windage * vx
    dvydt = -g - windage * vy
    return dxdt, dydt, dvxdt, dvydt


def rk4_step(x, y, vx, vy, dt):
    k1x, k1y, k1vx, k1vy = derivatives(x, y, vx, vy)
    k2x, k2y, k2vx, k2vy = derivatives(
        x + 0.5*dt*k1x,   y + 0.5*dt*k1y,
        vx + 0.5*dt*k1vx, vy + 0.5*dt*k1vy
    )
    k3x, k3y, k3vx, k3vy = derivatives(
        x + 0.5*dt*k2x,   y + 0.5*dt*k2y,
        vx + 0.5*dt*k2vx, vy + 0.5*dt*k2vy
    )
    k4x, k4y, k4vx, k4vy = derivatives(
        x + dt*k3x,   y + dt*k3y,
        vx + dt*k3vx, vy + dt*k3vy
    )
    x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    y += dt*(k1y + 2*k2y + 2*k3y + k4y)/6
    vx += dt*(k1vx + 2*k2vx + 2*k3vx + k4vx)/6
    vy += dt*(k1vy + 2*k2vy + 2*k3vy + k4vy)/6
    return x, y, vx, vy


def init_anim():
    traject_an.set_data(x_an, y_an)
    traject_num.set_data(x_num[:1], y_num[:1])
    return (traject_an, traject_num)


def loop_anim(i):
    global vx, vy

    x_num[i+1], y_num[i+1], vx, vy = rk4_step(
        x_num[i], y_num[i],
        vx, vy, dt
    )
    if y_num[i+1] <= 0:
        return (traject_an, traject_num)
    traject_num.set_data(x_num[:i+2], y_num[:i+2])
    return (traject_an, traject_num)

ani = animation.FuncAnimation(
    fig=fig, func=loop_anim, init_func=init_anim, 
    frames=nsteps_num, interval=10, repeat=False
)

plt.show()
