import numpy as np
import matplotlib.pyplot as plt


def simple_RK2(w, dt, steps):
    x = 0
    y_0 = 1
    y_1 = 0
    x_array = np.zeros((steps, 1))
    y_0_array = np.zeros((steps, 1))
    y_1_array = np.zeros((steps, 1))
    for i in range(0, steps):
        x_array[i] = x
        y_0_array[i] = y_0
        y_1_array[i] = y_1

        y_0 += y_1_array[i-1] * dt

        k1 = -w**2 * y_0_array[i-1]
        k2 = -w**2 * (y_0_array[i-1] + k1) * dt
        y_1 += (0.5 * (k1 + k2))
        x += dt

    return x_array, y_0_array


def simple_EulerCauchy(w, dt, steps):
    x = 0
    y_0 = 1
    y_1 = 0
    x_array = np.zeros((steps, 1))
    y_0_array = np.zeros((steps, 1))
    y_1_array = np.zeros((steps, 1))
    for i in range(0, steps):
        x_array[i] = x
        y_0_array[i] = y_0
        y_1_array[i] = y_1

        y_0 += y_1_array[i-1] * dt

        k1 = -w**2 * y_0_array[i-1]
        k2 = -w**2 * (y_0_array[i-1] + k1 * 0.5) * dt
        y_1 += k2
        x += dt

    return x_array, y_0_array


w_0 = 1
x0, y0 = simple_RK2(0.07, w_0, 100)
z = np.cos(w_0 * x0)
x1, y1 = simple_EulerCauchy(0.13, w_0, 100)

plt.plot(x0, z, 'r-')
plt.plot(x0, y0)
plt.plot(x1, y1)
plt.show()

