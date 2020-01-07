import numpy as np
import matplotlib.pyplot as plt


def forward_euler(dx, a, N):
    y = 1
    y_array = np.zeros((N, 1))
    for i in range(0, N):
        y_array[i] = y
        y *= (1 - a * dx)
    return y_array


def backward_euler(dx, a, N):
    y = 1
    y_array = np.zeros((N, 1))
    for i in range(0, N):
        y_array[i] = y
        y = y/(1 + a * dx)
    return y_array


alpha = 1

x = np.linspace(0, 10, 1000)
y = np.exp(-alpha*x)

dt = 1.6
y1 = forward_euler(dt, alpha, int(10/dt))
y2 = backward_euler(dt, alpha, int(10/dt))
x1 = np.linspace(0, 10, int(10/dt))

plt.plot(x, y, 'r-')
plt.plot(x1, y1)
plt.plot(x1, y2)
plt.show()

