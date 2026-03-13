import math
import numpy as np

g = 9.81
windage = 0.5

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


def solve_forward_problem(v0, grads):
    # PHYSICS
    alpha = math.radians(grads)

    vx = v0 * math.cos(alpha)
    vy = v0 * math.sin(alpha)
    x = 0.0
    y = 0.0
    rang = -1.0

    # NUMERIC
    nsteps_num = 40000
    dt = 0.01

    # ACTION LOOP
    for i in range(nsteps_num):
        x_prev, y_prev = x, y
        x, y, vx, vy = rk4_step(x, y, vx, vy, dt)
        if y <= 0:
            frac = y_prev / (y_prev - y)
            return x_prev + frac * (x - x_prev)
    return rang


print(solve_forward_problem(100, 45))
