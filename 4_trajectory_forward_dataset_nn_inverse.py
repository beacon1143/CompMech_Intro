import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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
        if y <= 0:  # linear interpolation for greater accuracy
            frac = y_prev / (y_prev - y)
            return x_prev + frac * (x - x_prev)
    return rang

def generate_dataset(N, v):
    print('\nDataset generation started...')
    print('Dataset size is', N)

    alpha = np.random.uniform(45.0, 89.95, N) # to make R(alpha) monotonic
    R = np.array([solve_forward_problem(v, al) for al in alpha])

    # MONOTONICITY CHECK
    #plt.scatter(alpha, R)
    #plt.xlabel("angle")
    #plt.ylabel("range")
    #plt.show()

    R_mean = R.mean()
    R_std = R.std()
    X = ((R - R_mean) / R_std).reshape(-1,1)

    alpha_mean = alpha.mean()
    alpha_std = alpha.std()
    y = (alpha - alpha_mean) / alpha_std

    print('Dataset generation completed!')

    return X, y, R_mean, R_std, alpha_mean, alpha_std


def create_and_train_nn(X, y, epochs_number):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print('\nNeural network created, training started...')
    print('Epochs number is', epochs_number)
    print('------------\nEpoch\tLoss')

    # TRAINING
    for epoch in range(epochs_number):
        pred = model(X_t)
        loss = loss_fn(pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(epoch, loss.item(), sep='\t')

    print('Neural network training completed!')

    # CHECK QUALITY
    #pred = model(X_t).detach().numpy().flatten()
    #theta_true = y * theta_std + theta_mean
    #theta_pred = pred * theta_std + theta_mean
    #plt.scatter(theta_true, theta_pred)
    #plt.plot([45,90],[45,90],'r')
    #plt.xlabel("true angle")
    #plt.ylabel("predicted angle")
    #plt.show()

    return model


def solve_inverse_problem(model, desired_range, R_mean, R_std, alpha_mean, alpha_std):
    R_test_n = (desired_range - R_mean) / R_std
    inp = torch.tensor([[R_test_n]], dtype=torch.float32)

    alpha_pred = model(inp).item()
    alpha_pred = alpha_pred * alpha_std + alpha_mean

    return alpha_pred

dataset_size = 10000
epochs_number = 5000

print('Initial velocity is fixed to', 100)
X, y, R_mean, R_std, alpha_mean, alpha_std = generate_dataset(dataset_size, 100)
model = create_and_train_nn(X, y, epochs_number)


for r in range(10, 131, 20):    
    print('\nTarget range is', r)
    alpha = solve_inverse_problem(model, r, R_mean, R_std, alpha_mean, alpha_std)
    print('Predicted angle is', alpha, 'degrees')
    print('True range for this angle is', solve_forward_problem(100, alpha))
