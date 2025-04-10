import math
import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
v0 = 10
alpha = math.radians(45.0)
g = 9.81
flight_range = v0 * v0 * math.sin(2 * alpha) / g

# NUMERIC
nsteps = 100

# PREPROCESSING
x = np.linspace(0, flight_range, nsteps + 1)
y_an = x * math.tan(alpha) - g * x * x / (2.0 * v0 * v0 * math.cos(alpha) ** 2)

# FIGURE
plt.plot(x, y_an)
plt.axis('equal')
plt.show()
