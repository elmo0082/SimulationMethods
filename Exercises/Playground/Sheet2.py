import numpy as np
import matplotlib.pyplot as plt

m1 = 0.5
m2 = 1.
l1 = 2.
l2 = 1.
g = 1.
h = 0.05
time = 100
steps = time / h

# v = (phi1, phi2, q1, q2)

# phi1 dot
def f1(v):
    return (m2 * l1**2 * v[2] - m2 * l1 * l2 * np.cos(v[0] - v[1]) * v[3]) / \
           (m2 * (m1 + m2) * l1**2 * l2**2 - m2**2 * l1**2 * l2**2 * np.cos(v[0] - v[1])**2)


# phi2 dot
def f2(v):
    return ((m1 + m2) * l1**2 * v[3] - m2 * l1 * l2 * np.cos(v[0] - v[1]) * v[2]) / \
           (m2 * (m1 + m2) * l1**2 * l2**2 - m2**2 * l1**2 * l2**2 * np.cos(v[0] - v[1])**2)


def f3(v):
    return -m2 * l1 * l2 * f1(v) * f2(v) * np.sin(v[0] - v[1]) - (m1 + m2) * g * l1 * np.sin(v[0])


def f4(v):
    return m2 * l1 * l2 * f1(v) * f2(v) * np.sin(v[0] - v[1]) - m2 * g * l2 * np.sin(v[1])


def f(v):
    f = np.zeros((4, 1))
    f[0] = f1(v)
    f[1] = f2(v)
    f[2] = f3(v)
    f[3] = f4(v)
    return f


def energy(v):
    return m1/2 * (l1 * f1(v))**2 + m2/2 * ((l1 * f1(v))**2 + (l2 * f2(v))**2 + 2 * l1 * l2 * f1(v) * f2(v) * np.cos(v[0] - v[1])) \
           + m1 * g * l1 * (1 - np.cos(v[0])) + m2 * g * (l1 * (1 - np.cos(v[0])) + l2 * (1 - np.cos(v[1])))


def RK2_step(v):
    k1 = f(v)[:] * h
    k2 = f(v + k1 * h) * h

    return v + 0.5 * (k1 + k2)


def RK4_step(v):
    k1 = f(v)[:] * h
    k2 = f(v + k1 * h/2) * h
    k3 = f(v + k2 * h/2) * h
    k4 = f(v + k3 * h) * h

    return v + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_algorithm(phi1, phi2, q1, q2):
    v = np.zeros((4, 1))
    v[0] = phi1
    v[1] = phi2
    v[2] = q1
    v[3] = q2

    E0 = energy(v)
    positions = np.zeros((int(steps), 4))
    Ev = np.zeros((int(steps), 4))
    Ew = np.zeros((int(steps), 4))
    w = v
    for i in range(0, int(steps)):
        print(i)
        v = RK4_step(v)
        w = RK2_step(w)

        Ev[i] = np.abs((energy(v) - E0) / E0)
        Ew[i] = np.abs((energy(w) - E0) / E0)

        plt.cla()
        #plt.axis([-3, 3, -3, 3])
        positions[i] = calculate_coordinates(v[0], v[1])

        pos = cast_positions(positions)
        #plt.plot(pos[0], pos[2])
        #plt.plot(pos[1], pos[3])

        #plt.savefig('./PendulumMovie/pendulum_' + frame_index(i) + '.png')

    return Ev, Ew


def calculate_coordinates(phi1, phi2):
    x1 = - l1 * np.sin(phi1)
    x2 = x1 - l2 * np.sin(phi2)
    y1 = -l1 * np.cos(phi1)
    y2 = y1 - l2 * np.cos(phi2)
    return [x1, x2, y1, y2]


def cast_positions(positions):
    coordinates = np.zeros((4, int(steps)))
    for i in range(0, int(steps)):
        coordinates[0][i] = positions[i][0]
        coordinates[1][i] = positions[i][1]
        coordinates[2][i] = positions[i][2]
        coordinates[3][i] = positions[i][3]
    return coordinates


def frame_index(i):
    # Command to create movie:
    # cat * | ffmpeg -framerate 8 -f image2pipe -i - -c:v copy video.mp4

    if i < 10:
        return "0000" + str(i)
    elif i < 100:
        return "000" + str(i)
    elif i < 1000:
        return "00" + str(i)
    elif i < 10000:
        return "0" + str(i)
    else:
        return str(i)


RK4, RK2 = run_algorithm(50 * (np.pi/180), -120 * (np.pi/180), 0, 0)
x = np.linspace(0, 100, 2000)
plt.plot(x, RK4)
plt.plot(x, RK2)
plt.plot(x, RK4 - RK2)
plt.show()

