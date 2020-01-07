import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

G = 0.1
h = 0.01
time = 20
steps = 13000  # time / h
N = 3

#m = np.ones((1, N))
#m[0][2] = 0.1
#x0 = np.array([[-0.5, 0.5, 1], [0, 0, 6], [0, 0, 2]])
#v0 = np.array([[0, 0, 0], [-0.5, 0.5, 0], [0, 0, 0]])


def generate_random_stars(N):
    m = np.ones((1, N))
    x = np.zeros((3, N))
    v = np.zeros_like(x)

    for i in range(0, N):
        x[0][i] = np.random.uniform(-1, 1)
        x[1][i] = np.random.uniform(-1, 1)
        x[2][i] = np.random.uniform(-1, 1)

        v[0][i] = np.random.uniform(-0.1, 0.1)
        v[1][i] = np.random.uniform(-0.1, 0.1)
        v[2][i] = np.random.uniform(-0.1, 0.1)

    return x, v, m


x0, v0, m = generate_random_stars(N)
print(x0)


def distance(x, y):
    d = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2, (x[2] - y[2]) ** 2)
    d[d <= 0] = float('Inf')
    return d


def a(x):
    accelerations = np.zeros_like(x)
    for i in range(0, N):
        X = np.array([x[0, i], x[1, i], x[2, i]])
        v = np.array([x[0][i] - x[0], x[1][i] - x[1], x[2][i] - x[2]])
        accelerations += G * m[0] * v / (distance(X, x)**3)
    return accelerations


def leap_frog_step(x, v, h):
    x = x + v * h
    v = v + a(x) * h
    return x, v


def run_leap_frog(x, v, h):
    positions = np.zeros((int(steps), 3, N))
    for i in range(0, int(steps)):
        positions[i] = x
        x, v = leap_frog_step(x, v, h)
    return positions


def run_leap_frog_close_encounter(x, v, h):
    positions = np.zeros((int(steps), 3, N))
    H = h
    T = 0
    for i in range(0, int(steps)):
        T += H
        positions[i] = x
        x, v = leap_frog_step(x, v, H)
        H = close_encounter_checker(x, h, N)
    print(T)
    return positions


def plot_result(array):
    for i in range(0, N):
        s = 3
        x = array[:, 0, i]
        y = array[:, 1, i]
        plt.axis([-s, s, -s, s])
        plt.plot(x, y)
    plt.show()


def close_encounter_checker(x, h, N):
    for i in range(0, N):
        X = np.array([x[0][i], x[1][i], x[2][i]])
        if any(distance(X, x) < 0.1):
            print('Close encounter!')
            return h * 0.001
        else:
            return h


plot_result(run_leap_frog_close_encounter(x0, v0, h))


