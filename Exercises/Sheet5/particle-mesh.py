import numpy as np
import matplotlib.pyplot as plt

# Particle Mesh of Width 30 with 30 x 30 cells
H = 15
K = 30
M = 2.0
N = 100
sigma = 4
particles = []


# Find the index of a given point
def indices(X, Y):
    k = np.floor(X)
    j = np.floor(Y)
    return [int(k), int(j)]


# Places particles within the interval [-14, 14)
def insert_particles(h, size):
    x = 2 * (h - 1) * (np.random.random_sample(size)) + 1
    y = 2 * (h - 1) * (np.random.random_sample(size)) + 1
    return list(zip(x.tolist(), y.tolist()))


# Do the same with gaussian distribution
def insert_gaussian_particles(h, size):
    x = []
    y = []
    for i in range (size):
        x0 = np.random.normal(loc=14.5, scale=sigma)
        y0 = np.random.normal(loc=14.5, scale=sigma)
        if x0 > 29 or x0 < 1 or y0 > 29 or y0 < 1:
            i -= 1
        else:
            x.append(x0)
            y.append(y0)
    return list(zip(x, y))


# calculate the values of the epsilons
def calculate_e(X, index):
    return np.abs((X - index) / (2*H/K))


def compute_0_order(X, Y, k, l):
    W = np.zeros((3, 3))
    W[1, 1] = 1
    return W


def compute_1_order(X, Y, k, l):
    W = np.ones((3, 3))
    ex = calculate_e(X, k)
    ey = calculate_e(Y, l)
    if ex < 0.5:
        W[:, 0] *= H/K - ex
        W[:, 1] *= ex + H/K
        W[:, 2] *= 0
    else:
        W[:, 0] *= 0
        W[:, 1] *= 3*H/K - ex
        W[:, 2] *= ex - H/K

    if ey < 0.5:
        W[0, :] *= H/K - ey
        W[1, :] *= ey + H/K
        W[2, :] *= 0
    else:
        W[0, :] *= 0
        W[1, :] *= 3*H/K - ey
        W[2, :] *= ey - H/K
    return W


def compute_2_order(X, Y, k, l):
    ex = calculate_e(X, k)
    ey = calculate_e(Y, l)
    W = np.ones((3, 3))
    W[:, 0] *= 0.5 - ex + 0.5*ex**2
    W[:, 1] *= 0.5 + ex - ex ** 2
    W[:, 2] *= 0.5 * ex**2
    W[0, :] *= 0.5 - ey + 0.5 * ey ** 2
    W[1, :] *= 0.5 + ey - ey ** 2
    W[2, :] *= 0.5 * ey ** 2
    return W


def compute_order(X, Y, k, l, order):
    if order == 0:
        return compute_0_order(X, Y, k, l)
    if order == 1:
        return compute_1_order(X, Y, k, l)
    if order == 2:
        return compute_2_order(X, Y, k, l)


def density_map(particle_list, order):
    d_map = np.zeros((K, K))
    for k in range(1, K-1):
        for j in range(1, K-1):
            for particle in particle_list:
                x, y = particle[0], particle[1]  # particle position
                if indices(x, y) == [k, j]:
                    W = compute_order(x, y, k, j, order)
                    d_map[np.ix_([j-1, j, j+1], [k-1, k, k+1])] += M/N * W
    return d_map


particles = insert_gaussian_particles(H, N)

print(np.sum(density_map(particles, order=0)))
print(np.sum(density_map(particles, order=1)))
print(np.sum(density_map(particles, order=2)))

fig, ax = plt.subplots()

plt.imshow(density_map(particles, order=0), cmap=plt.cm.Blues, origin='lower', interpolation='nearest')
plt.colorbar()
#for i in range(len(particles)):
#    ax.plot(particles[i][0] - H / K, particles[i][1] - H / K, 'rx')

plt.xlabel("x, k")
plt.ylabel("y,l")
plt.grid()

fig, ax = plt.subplots()

plt.imshow(density_map(particles, order=1), cmap=plt.cm.Blues, origin='lower', interpolation='nearest')
plt.colorbar()
#for i in range(len(particles)):
#    ax.plot(particles[i][0] - H / K, particles[i][1] - H / K, 'rx')

plt.xlabel("x, k")
plt.ylabel("y,l")
plt.grid()

fig, ax = plt.subplots()

plt.imshow(density_map(particles, order=2), cmap=plt.cm.Blues, origin='lower', interpolation='nearest')
plt.colorbar()
#for i in range(len(particles)):
#   ax.plot(particles[i][0] - H / K, particles[i][1] - H / K, 'rx')

plt.xlabel("x, k")
plt.ylabel("y,l")
plt.grid()

plt.show()

