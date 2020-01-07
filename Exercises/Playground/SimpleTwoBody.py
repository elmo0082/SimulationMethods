import numpy as np
import matplotlib.pyplot as plt

Ms = 1  # 1.99e30
Mp = 1e-3 * Ms
AU = 1  # 1.496e11
vk = 1  # 2.98e4
G = 1  # 6.6743e-11
h = 0.02
time = 20
steps = time / h


def a(s):
    return -G * Ms * s / np.sqrt(s[0]**2 + s[1]**2)**3


def f(v):
    return np.array([v[1], a(v[0])])


def leap_frog_step(s, w):
    s = s + w * h
    w = w + a(s) * h
    return s, w


def RK2_step(s, w):
    v = np.array([s, w])
    k1 = f(v) * h
    k2 = f(v + k1) * h
    v = v + 0.5 * (k1 + k2)
    return v


def run_leap_frog(s, w):
    positions = np.zeros((2, int(steps)))
    for i in range(0, int(steps)):
        positions[0][i] = s[0]
        positions[1][i] = s[1]
        s, w = leap_frog_step(s, w)
    return positions


def run_RK2(s, w):
    positions = np.zeros((2, int(steps)))
    for i in range(0, int(steps)):
        positions[0][i] = s[0]
        positions[1][i] = s[1]
        s, w = RK2_step(s, w)
    return positions


s0 = np.array([1, 0]) * AU
w0 = np.array([0, 0.5]) * vk

leappos = run_leap_frog(s0, w0)
RK2pos = run_RK2(s0, w0)
plt.plot(leappos[0], leappos[1])
plt.plot(RK2pos[0], RK2pos[1])
plt.plot(0, 0, 'ko')
plt.show()

